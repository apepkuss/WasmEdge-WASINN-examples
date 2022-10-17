use image;
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::Read;
use wasmedge_nn::nn;
mod imagenet_classes;

pub fn main() {
    main_entry();
}

#[no_mangle]
fn main_entry() {
    infer_image();
}

fn infer_image() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    let image_name: &str = &args[2];

    // load image and convert it to tensor
    println!("Load image file and convert it into tensor ...");
    let bytes = image_to_tensor(image_name.to_string(), 224, 224);
    let tensor = nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        type_: nn::Dtype::F32.into(),
        data: bytes.as_slice(),
    };

    // create wasi-nn context
    let mut ctx = nn::ctx::WasiNnCtx::new()?;

    println!("Load model files ...");
    let graph_id = ctx.load(
        None,
        model_bin_name,
        nn::GraphEncoding::Pytorch,
        nn::ExecutionTarget::CPU,
    )?;
    println!("Loaded graph into wasi-nn with ID: {}", graph_id);

    println!("initialize the execution context ...");
    let exec_context_id = ctx.init_execution_context(graph_id)?;
    println!(
        "Created wasi-nn execution context with ID: {}",
        exec_context_id
    );

    println!("Set input tensor ...");
    ctx.set_input(exec_context_id, 0, tensor)?;

    println!("Do inference ...");
    ctx.compute(exec_context_id)?;

    println!("Extract result ...");
    let mut out_buffer = vec![0u8; 1000 * 4];
    ctx.get_output(exec_context_id, 0, out_buffer.as_mut_slice())?;

    let mut buffer = vec![0f32; 1000];
    for i in 0..1000 {
        // let x: &[u8; 4] = &out_buffer[i * 4..(i + 1) * 4].try_into().unwrap();
        buffer.push(f32::from_le_bytes(
            out_buffer[i * 4..(i + 1) * 4].try_into().unwrap(),
        ));
    }

    let results = sort_results(&buffer);
    for i in 0..5 {
        println!(
            "   {}.) [{}]({:.4}){}",
            i + 1,
            results[i].0,
            results[i].1,
            imagenet_classes::IMAGENET_CLASSES[results[i].0]
        );
    }

    Ok(())
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&img, height, width, ::image::imageops::FilterType::Triangle);
    let mut flat_img: Vec<f32> = Vec::new();
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }
    let bytes_required = flat_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for c in 0..3 {
        for i in 0..(flat_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = flat_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((flat_img.len() / 3 * c + i) * 4) + j] = u8_bytes[j];
            }
        }
    }
    return u8_f32_arr;
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
