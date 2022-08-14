use ndarray::prelude::*;
use rand::prelude::SliceRandom;

use crate::{Input, Target, CLASS};

// Splits the data into (train, test) sets.
pub fn _split_data(input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
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

pub fn split_multiclass(input: &Target, classes: usize) -> Array1<Vec<usize>> {
    let mut arr = Array1::<Vec<usize>>::from_elem(classes, Vec::new());

    // Loop through all inputs.
    for (i, entry) in input.iter().enumerate() {
        // Check if that input matches any of the classes.    
        for (c, class) in CLASS.iter().enumerate() {
            if entry == class {
                // Add to the correct vector if so.
                arr[c].push(i);
            }
        }
        
    }

    arr
}

pub fn split_multiclass_input(input: Input, classes: usize) -> Input {
    let arr = split_multiclass(&input.1, classes);

    let mut data = Vec::new();
    let mut target = Vec::new();

    for class in arr {
        for index in class {
            target.push(input.1[index].to_owned());
            for i in input.0.row(index) {
                data.push(*i);
            }
        }
    }

    (Array2::<f32>::from_shape_vec(input.0.raw_dim(), data).unwrap(), 
     Array1::<String>::from_shape_vec(input.1.raw_dim(), target).unwrap())
}
