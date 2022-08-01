/*
Programming Assignment #2 - Naiive Bayes Classification

Name: Thomas Abel
Date: 2022-08-01
Class: Machine Learning
*/
mod read;
mod print_data;
mod model;

use ndarray::prelude::*;
use print_data::*;
use rand::prelude::SliceRandom;

fn main() {
    // Read data file.
    let path_index = 0;
    let path = [
        "./data/spambase.data"
    ];
    println!("Reading data, please be patient...");
    let result = read::read_csv(path[path_index]);
    let input;
    match result {
        Ok(o) => {
            println!("SUCCESS: Data read \n");
            input = o;
        },
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    let temp = split_data(&input);
    let train = temp.0;
    let test = temp.1;

    let model = model::Bayes::train(&train);
    let confusion = model.test(&test);
    print_confusion(&confusion);

    println!("Ending session.");
}

// Splits the data into (train, test) sets.
fn split_data(input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let mut false_set = Vec::new();
    let mut true_set = Vec::new();

    // Parse each row as true or false and store indecies.
    for i in 0..input.len_of(Axis(0)) {
        if *input.row(i).last().unwrap() == 0. {
            false_set.push(i);
        }
        else {
            true_set.push(i);
        }
    }

    // Shuffle the sets.
    let mut rng = rand::thread_rng();
    false_set.shuffle(&mut rng);
    true_set.shuffle(&mut rng);
    
    // Get lengths of each set.
    let f_len = false_set.len();
    let t_len = true_set.len();
    
    // Split sets in half.
    let train = [&false_set[0..f_len/2], &true_set[0..t_len/2]].concat();
    let test = [&false_set[f_len/2..f_len], &true_set[t_len/2..t_len]].concat();

    // Copy data into new arrays.
    let columns = input.len_of(Axis(1));
    let mut rows = train.len();
    let mut train_set = Array2::<f32>::zeros((rows, columns));
    for i in 0..rows {
        for j in 0..columns {
            train_set[[i, j]] = input[[train[i], j]];
        }
    }
    rows = test.len();
    let mut test_set = Array2::<f32>::zeros((rows, columns));
    for i in 0..rows {
        for j in 0..columns {
            test_set[[i, j]] = input[[test[i], j]];
        }
    }

    (train_set, test_set)
}

// Confusion matrix functions
fn accuracy(confusion: &Array2<u32>) -> f32 {
    (confusion[[0, 0]] + confusion[[1, 1]]) as f32 / confusion.sum() as f32
}
fn precision(confusion: &Array2<u32>) -> f32 {
    confusion[[1, 1]] as f32 / (confusion[[0, 0]] + confusion[[1, 1]]) as f32
}
fn recall(confusion: &Array2<u32>) -> f32 {
    confusion[[1, 1]] as f32 / (confusion[[1, 1]] + confusion[[1, 0]]) as f32
}
fn print_confusion(confusion: &Array2<u32>) {
    _print_matrix(&confusion.view(), "CONFUSION");
    println!("Accuracy:  {:<12.6}", accuracy(confusion));
    println!("Precision: {:<12.6}", precision(confusion));
    println!("Recall:    {:<12.6}", recall(confusion));
    println!();
}