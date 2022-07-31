/*
Programming Assignment #2 - Naiive Bayes Classification

Name: Thomas Abel
Date: 2022-08-01
Class: Machine Learning
*/
mod constants;
mod read;
mod print_data;

use ndarray::prelude::*;
use print_data::*;
use rand::prelude::SliceRandom;
use crate::constants::*;

fn main() {
    // Read data file.
    let path_index = 0;
    let path = [
        "./data/spambase.data"
    ];
    let result = read(path[path_index]);
    let input;
    match result {
        Ok(o) => {
            input = o;
        },
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    // Change inputs here:
    let temp = split_data(&input);
    let training_set = temp.0;
    let testing_set = temp.1;
    
    _print_matrix(&testing_set.view(), "Test");

    println!("Number of");
    println!("Inputs: {}", input.len_of(Axis(0)));
    println!("Train Set: {}", training_set.len_of(Axis(0)));
    println!("Test Set: {}", testing_set.len_of(Axis(0)));
    println!("Ending session.");
}

fn read(path: &str) -> Result<Array2<f32>, &str> {
    println!("Reading data, please be patient...");
    let input 
        = read::read_csv(path);
    match input {
        Ok(v) => {
            println!("SUCCESS: Data read");
            Ok(v)
        }
        Err(_e) => {
            let str = "Could not read.";
            Err(str)
        }
    }
}

fn split_data(
    input: &Array2<f32>, 
) -> (Array2<f32>, Array2<f32>) {
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
    let mut train = [&false_set[0..f_len/2], &true_set[0..t_len/2]].concat();
    let mut test = [&false_set[f_len/2..f_len], &true_set[t_len/2..t_len]].concat();

    // Shuffle again.
    train.shuffle(&mut rng);
    test.shuffle(&mut rng);

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
