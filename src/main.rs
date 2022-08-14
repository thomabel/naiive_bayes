/*
Programming Assignment #2 - Naiive Bayes Classification

Name: Thomas Abel
Date: 2022-08-01
Class: Machine Learning
*/
mod utility;
mod model;
mod model_mnist;

use ndarray::prelude::*;
use std::error::Error;
use crate::utility::*;

type Vector = Array1<f32>;
type Matrix = Array2<f32>;
type Confusion = Array2<u32>;
type Target = Array1<String>;
type Input = (Matrix, Target);
type ReadResult = Result<Input, Box<dyn Error>>;
type MeanStd = Array1<(f32, f32)>;
type MeanStd2 = Array2<(f32, f32)>;

pub const CLASS: [&str; 10] = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ];

fn main() {
    // Read data file.
    let train_index = 0;
    let test_index = 0;
    let path = [
        "./data/mnist_test_short.csv",    
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
        "./data/spambase.data",
    ];
    let divisor = 1.;
    let classes = CLASS.len();
    let train_input 
        = split::split_multiclass_input(read::read_eval(path[train_index], divisor), classes);
    let test_input 
        = split::split_multiclass_input(read::read_eval(path[test_index], divisor), classes);

    let model = model_mnist::Bayes::train(&train_input, &CLASS);
    let confusion = model.test(&test_input, &CLASS);
    print_confusion(&confusion);

    println!("Ending session.");
}


// Confusion matrix functions
fn print_confusion(confusion: &Confusion) {
    print_data::_print_matrix(&confusion.view(), "CONFUSION");
    println!("Accuracy:  {:<12.6}", accuracy(confusion));
    println!();
}
fn accuracy(confusion: &Confusion) -> f32 {
    let correct = confusion.diag().sum() as f32;
    let total = confusion.sum() as f32;
    correct / total
}
