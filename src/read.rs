use std::error::Error;
use ndarray::prelude::*;

pub fn read_csv (path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut output = Vec::new();
    let mut rows: usize = 0;
    let mut columns: usize = 0;

    // Parse each row in the input.
    for result in reader.records() {
        let record = result?;
        columns = record.len();

        // Parse each entry in that row as an f32.
        for r in record.into_iter() {
            let num = r.parse::<f32>();
            match num {
                Ok(c) => {
                    output.push(c);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        rows += 1;
    }
    let out_array 
        = Array2::from_shape_vec((rows, columns), output)?;
    Ok(out_array)
}