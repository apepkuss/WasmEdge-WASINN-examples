use std::convert::TryInto;
use std::env;
use std::fs;
use wasi_nn;
// mod imagenet_classes;

pub fn main() {
    main_entry();
}

#[no_mangle]
fn main_entry() {
    infer_image();
}

fn infer_image() {
    let args: Vec<String> = env::args().collect();
    let model_xml_name: &str = &args[1];
    let model_bin_name: &str = &args[2];
    let tensor_name: &str = &args[3];

    let xml = fs::read_to_string(model_xml_name).unwrap();
    println!("Read graph XML, size in bytes: {}", xml.len());

    let weights = fs::read(model_bin_name).unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let graph = unsafe {
        wasi_nn::load(
            &[&xml.into_bytes(), &weights],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
    println!("Created wasi-nn execution context with ID: {}", context);

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    let tensor_data = fs::read(tensor_name).unwrap();
    println!("Read input tensor, size in bytes: {}", tensor_data.len());
    // for i in 0..10{
    //     println!("tensor -> {}", tensor_data[i]);
    // }
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };
    unsafe {
        wasi_nn::set_input(context, 0, tensor).unwrap();
    }
    // Execute the inference.
    unsafe {
        wasi_nn::compute(context).unwrap();
    }
    println!("Executed graph inference");
    // Retrieve the output.
    let mut output_buffer = vec![0f32; 1001];
    unsafe {
        wasi_nn::get_output(
            context,
            0,
            &mut output_buffer[..] as *mut [f32] as *mut u8,
            (output_buffer.len() * 4).try_into().unwrap(),
        )
        .unwrap();
    }

    // let results = sort_results(&output_buffer);
    // for i in 0..5 {
    //     println!(
    //         "   {}.) [{}]({:.4}){}",
    //         i + 1,
    //         results[i].0,
    //         results[i].1,
    //         imagenet_classes::IMAGENET_CLASSES[results[i].0]
    //     );
    // }
    // let ground_truth_result = [963, 762, 909, 926, 567];
    // // let ground_truth_pred = [0.7113048, 0.0707076, 0.036355935, 0.015456136, 0.015344063];
    // for i in 0..ground_truth_result.len() {
    //     assert_eq!(results[i].0, ground_truth_result[i]);
    // }
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);

// use std::convert::TryInto;
// use std::fs;
// use wasi_nn;
// // mod imagenet_classes;
// use anyhow::{anyhow, Result};
// use opencv::{
//     self as cv,
//     core::{
//         MatExprResult, MatTraitConstManual, Scalar, Size, ToInputArray, VecN, VectorElement, CV_32F,
//     },
//     dnn::blob_from_image,
//     imgcodecs::{imread, ImreadModes},
//     imgproc::{resize, InterpolationFlags},
//     prelude::*,
// };
// use std::path::Path;

// use image::io::Reader;
// use image::DynamicImage;

// pub fn main() -> Result<()> {
//     main_entry()?;

//     Ok(())
// }

// #[no_mangle]
// fn main_entry() -> Result<()> {
//     let model_dir = std::env::current_dir()?
//         .parent()
//         .ok_or(anyhow!("Failed to change to parent directory"))?
//         .join("model");
//     assert!(model_dir.exists());

//     let bgr_file = Path::new("/Users/sam/workspace/python/dog_1x3x416x416.bgr");
//     let model_xml_file = model_dir.join("yolo-v3-onnx.xml");
//     let model_bin_file = model_dir.join("yolo-v3-onnx.bin");

//     let tensor_data = fs::read(tensor_name).unwrap();
//     println!("Read input tensor, size in bytes: {}", tensor_data.len());

//     // preprocess image
//     // let image_bytes = preprocess(image_file)?;

//     // infer
//     infer(model_xml_file, model_bin_file, image_bytes)?;

//     // todo: post-process

//     Ok(())
// }

// // fn process_image(filename: impl AsRef<str>) -> Result<Vec<u8>> {
// //     // read image from a file and convert it to 3 channel BGR color image
// //     let frame = imread(filename.as_ref(), ImreadModes::IMREAD_COLOR as i32)?;

// //     // resize image to 1x3x416x416
// //     let blob = blob_from_image(
// //         &frame,
// //         1.0 / 255.0,
// //         Size::new(416, 416),
// //         Scalar::new(0.0, 0.0, 0.0, 0.0),
// //         true,
// //         false,
// //         CV_32F,
// //     )?;

// //     Ok(cv_mat_to_tensor(blob)?)
// // }

// // fn cv_mat_to_tensor(blob: Mat) -> Result<Vec<u8>> {
// //     let raw_u8_arr = Mat::data_bytes(&blob)?;
// //     // Create an array to hold the f32 value of those pixels
// //     let bytes_required = raw_u8_arr.len() * 4;
// //     let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

// //     for i in 0..raw_u8_arr.len() {
// //         // Read the number as a f32 and break it into u8 bytes
// //         let u8_f32: f32 = raw_u8_arr[i] as f32;
// //         let u8_bytes = u8_f32.to_ne_bytes();

// //         for j in 0..4 {
// //             u8_f32_arr[(i * 4) + j] = u8_bytes[j];
// //         }
// //     }

// //     Ok(u8_f32_arr)
// // }

// // fn infer_image<P, T>(xml_file: P, bin_file: P, data: T) -> Result<()>
// // where
// //     P: AsRef<Path>,
// //     T: AsRef<[u8]>,
// // {
// //     let xml = fs::read_to_string(xml_file.as_ref()).unwrap();
// //     println!("Read graph XML, size in bytes: {}", xml.len());

// //     let weights = fs::read(bin_file.as_ref()).unwrap();
// //     println!("Read graph weights, size in bytes: {}", weights.len());

// //     let graph = unsafe {
// //         wasi_nn::load(
// //             &[&xml.into_bytes(), &weights],
// //             wasi_nn::GRAPH_ENCODING_OPENVINO,
// //             wasi_nn::EXECUTION_TARGET_CPU,
// //         )
// //         .unwrap()
// //     };
// //     println!("Loaded graph into wasi-nn with ID: {}", graph);

// //     let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
// //     println!("Created wasi-nn execution context with ID: {}", context);

// //     // Load a tensor that precisely matches the graph input tensor (see
// //     // `fixture/frozen_inference_graph.xml`).
// //     // let tensor_data = cv_mat_to_tensor(blob)?;
// //     let tensor_data = data.as_ref();
// //     println!("Read input tensor, size in bytes: {}", tensor_data.len());
// //     // for i in 0..10{
// //     //     println!("tensor -> {}", tensor_data[i]);
// //     // }
// //     let tensor = wasi_nn::Tensor {
// //         dimensions: &[1, 3, 416, 416],
// //         r#type: wasi_nn::TENSOR_TYPE_F32,
// //         data: tensor_data,
// //     };
// //     unsafe {
// //         wasi_nn::set_input(context, 0, tensor).unwrap();
// //     }
// //     // Execute the inference.
// //     unsafe {
// //         wasi_nn::compute(context).unwrap();
// //     }
// //     println!("Executed graph inference");
// //     // Retrieve the output.
// //     let mut output_buffer = vec![0f32; 80];
// //     unsafe {
// //         wasi_nn::get_output(
// //             context,
// //             0,
// //             &mut output_buffer[..] as *mut [f32] as *mut u8,
// //             (output_buffer.len() * 4).try_into().unwrap(),
// //         )
// //         .unwrap();
// //     }

// //     let results = sort_results(&output_buffer);
// //     dbg!(&results);
// //     // for i in 0..5 {
// //     //     println!(
// //     //         "   {}.) [{}]({:.4}){}",
// //     //         i + 1,
// //     //         results[i].0,
// //     //         results[i].1,
// //     //         // imagenet_classes::IMAGENET_CLASSES[results[i].0]
// //     //     );
// //     // }
// //     // let ground_truth_result = [963, 762, 909, 926, 567];
// //     // // let ground_truth_pred = [0.7113048, 0.0707076, 0.036355935, 0.015456136, 0.015344063];
// //     // for i in 0..ground_truth_result.len() {
// //     //     assert_eq!(results[i].0, ground_truth_result[i]);
// //     // }

// //     Ok(())
// // }

// // Sort the buffer of probabilities. The graph places the match probability for each class at the
// // index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// // to a wrapping InferenceResult and sort the results.
// fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
//     let mut results: Vec<InferenceResult> = buffer
//         .iter()
//         .skip(1)
//         .enumerate()
//         .map(|(c, p)| InferenceResult(c, *p))
//         .collect();
//     results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//     results
// }

// // A wrapper for class ID and match probabilities.
// #[derive(Debug, PartialEq)]
// struct InferenceResult(usize, f32);

// fn infer<P, T>(xml_file: P, bin_file: P, bytes: T) -> Result<()>
// where
//     P: AsRef<Path>,
//     T: AsRef<[u8]>,
// {
//     let xml = fs::read_to_string(xml_file.as_ref())?;
//     println!("Read graph XML, first 50 characters: {}", &xml[..50]);

//     let weights = fs::read(bin_file.as_ref())?;
//     println!("Read graph weights, size in bytes: {}", weights.len());

//     let graph = unsafe {
//         wasi_nn::load(
//             &[&xml.into_bytes(), &weights],
//             wasi_nn::GRAPH_ENCODING_OPENVINO,
//             wasi_nn::EXECUTION_TARGET_CPU,
//         )
//         .unwrap()
//     };
//     println!("Loaded graph into wasi-nn with ID: {}", graph);

//     let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
//     println!("Created wasi-nn execution context with ID: {}", context);

//     let filename =
//         Path::new("/Volumes/Dev/secondstate/me/WasmEdge-WASINN-examples/yolo-v3/rust/dog.jpeg");
//     let tensor_data = image_to_tensor(filename, 416, 416);
//     println!("Read input tensor, size in bytes: {}", tensor_data.len());
//     let tensor = wasi_nn::Tensor {
//         dimensions: &[1, 3, 416, 416],
//         r#type: wasi_nn::TENSOR_TYPE_F32,
//         data: &tensor_data,
//     };

//     // set input
//     unsafe {
//         wasi_nn::set_input(context, 0, tensor).unwrap();
//     };

//     // Execute the inference.
//     unsafe {
//         wasi_nn::compute(context).unwrap();
//     }
//     println!("Executed graph inference");

//     // Retrieve the output.
//     let mut output_buffer = vec![0f32; 80];
//     unsafe {
//         wasi_nn::get_output(
//             context,
//             0,
//             &mut output_buffer[..] as *mut [f32] as *mut u8,
//             (output_buffer.len() * 4).try_into().unwrap(),
//         )
//         .unwrap();
//     }

//     // let results = sort_results(&output_buffer);
//     // println!(
//     //     "Found results, sorted top 5: {:?}",
//     //     &results[..5]
//     // );

//     // for i in 0..5 {
//     //     println!("{}.) {}", i + 1, imagenet_classes::IMAGENET_CLASSES[results[i].0]);
//     // }

//     Ok(())
// }

// fn preprocess(image_file: impl AsRef<Path>, height: u32, width: u32) -> Result<Vec<u8>> {
//     let pixels = Reader::open(image_file.as_ref()).unwrap().decode().unwrap();
//     let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
// }

// // Take the image located at 'path', open it, resize it to height x width, and then converts
// // the pixel precision to FP32. The resulting BGR pixel vector is then returned.
// fn image_to_tensor(image_file: impl AsRef<Path>, height: u32, width: u32) -> Vec<u8> {
//     let pixels = Reader::open(image_file.as_ref()).unwrap().decode().unwrap();
//     let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
//     let bgr_img = dyn_img.to_bgr8();
//     // Get an array of the pixel values
//     let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
//     // Create an array to hold the f32 value of those pixels
//     let bytes_required = raw_u8_arr.len() * 4;
//     let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

//     for i in 0..raw_u8_arr.len() {
//         // Read the number as a f32 and break it into u8 bytes
//         let u8_f32: f32 = raw_u8_arr[i] as f32;
//         let u8_bytes = u8_f32.to_ne_bytes();

//         for j in 0..4 {
//             u8_f32_arr[(i * 4) + j] = u8_bytes[j];
//         }
//     }
//     return u8_f32_arr;
// }
