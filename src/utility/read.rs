use ndarray::prelude::*;
use csv::Reader;
use crate::{Input, ReadResult};

pub fn read_csv(path: &str) -> ReadResult {
    let mut reader = Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut target = Vec::new();
    let mut rows = 0;
    let columns = reader.headers()?.len() - 1;

    for r in reader.records() {
        let entry = r?;
        target.push(entry[0].to_string());
        for e in 1..entry.len() {
            let parse = entry[e].parse::<f32>();
            match parse {
                Ok(o) => {
                    output.push(o);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        rows += 1;
    }
    
    let output = 
        Array2::<f32>::from_shape_vec((rows, columns), output)?;
    let target = 
        Array1::<String>::from_shape_vec(rows, target)?;
    Ok( (output, target) )
}

pub fn read_eval(path: &str, divisor: f32) -> Input {
    evaluate(read_csv(path), divisor)
}

fn evaluate(result: ReadResult, divisor: f32) -> Input {
    let input;
    match result {
        Ok(mut o) => {
            // Normalize the data.
            o.0 /= divisor;
            input = o;
        }
        Err(e) => {
            panic!("{}", e);
        }
    }
    input
}
