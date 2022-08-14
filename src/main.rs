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
type MeanStd2 = Array2<(f32, f32)>;

pub const CLASS: [&str; 10] = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ];

fn main() {
    // Read data file.
    let train_index = 0;
    let test_index = 1;
    let path = [
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
        "./data/mnist_test_short.csv",    
        "./data/spambase.data",
    ];
    let divisor = 1.;
    let train_input = read::read_eval(path[train_index], divisor);
    let test_input = read::read_eval(path[test_index], divisor);

    let fraction_set = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ];
    for fraction in fraction_set {
        experiment(&train_input, &test_input, &CLASS, fraction);
    }

    println!("Ending session.");
}

// Trains a network using some fraction of the training data.
fn experiment(train_input: &Input, test_input: &Input, classes: &[&str], fraction: f32) {
    let class_len = classes.len();
    let train_input_sort 
        = split::split_multiclass_input(train_input, class_len, fraction);

    let model = model_mnist::Bayes::train(&train_input_sort, classes);
    let confusion = model.test(test_input, classes);
    print_confusion(&confusion);
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
