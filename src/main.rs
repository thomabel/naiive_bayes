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
use crate::constants::*;

fn main() {
    // Read data file.
    let path_index = 0;
    let path = [
        "./data/spambase.data"
    ];
    let result = read(path[path_index], INPUT);
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
    _print_matrix(&input.view(), "INPUT");
    
    println!("Ending session.");
}

fn read(path: &str, inputs: usize) -> Result<Array2<f32>, &str> {
    println!("Reading data, please be patient...");
    let input 
        = read::read_csv(path, inputs);
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