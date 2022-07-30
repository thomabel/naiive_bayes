use std::error::Error;
use ndarray::prelude::*;

pub fn read_csv (path: &str, inputs: usize) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut nrow = 0;

    // Parse each entry in the input.
    for result in reader.records() {
        let record = result?;
        for i in 0..inputs {
            let num = record[i].parse::<f32>();
            match num {
                Ok(c) => {
                    output.push(c);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
    
        nrow += 1;
    }

    let out_array 
        = Array2::from_shape_vec((nrow, inputs), output)?;
    Ok(out_array)
}