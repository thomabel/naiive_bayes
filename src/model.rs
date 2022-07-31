use ndarray::prelude::*;
use crate::constants::*;

pub struct Bayes {
    pub true_ratio: f64,
    pub false_ratio: f64,
    pub true_mean_std: Array1<(f32, f32)>,
    pub false_mean_std: Array1<(f32, f32)>,
}

impl Bayes {
    // Creates a new instance of this struct given an input data set.
    pub fn train(input: &Array2<f32>) -> Bayes {
        // Get information.
        let input_len = input.len_of(Axis(0));
        let input_true = Bayes::count(input, true);
            
        // The training set is split into true/false sections. This finds the line.
        let mut index = 0;
        for i in 0..input_len {
            if *input.row(i).last().unwrap() == 1. {
                index = i;
            }
        }
            
        // Slice the training set into true and false slices.
        let false_view = input.slice(s![0..index, ..]);
        let true_view = input.slice(s![index..input_len, ..]);
            
        // Calculate all necessary values for classification and return.
        Bayes {
            true_ratio: input_true as f64 / input_len as f64,
            false_ratio: (input_len - input_true) as f64 / input_len as f64,
            true_mean_std: Bayes::mean_std_dev(&true_view),
            false_mean_std: Bayes::mean_std_dev(&false_view),
        }
    }

    // Counts the number of true or false entries.
    fn count(input: &Array2<f32>, check: bool) -> usize {
        let mut count = 0;
        if check {
            for i in input.rows() {
                if *i.last().unwrap() == 1. {
                    count += 1;
                }
            }
        }
        else {
            for i in input.rows() {
                if *i.last().unwrap() == 0. {
                    count += 1;
                }
            }
        }
        count
    }
    
    // Calculates the mean and standard deviation of the columns from input.
    fn mean_std_dev(input: &ArrayView2<f32>) -> Array1<(f32, f32)> {
        let len = input.len_of(Axis(1));
        let mut output = Array1::<(f32, f32)>::from_elem(len, (0., 0.));
    
        for i in 0..len {
            let column = input.column(i);
            let mean = column.mean().unwrap();
            let c = column.std(0.);
            let std = 
            if c == 0. { 
                MIN_STD 
            }
            else {
                c
            };
            
            output[i] = (mean, std);
        }
    
        output
    }
    
    //=============================================================================================

    // Uses the test set and makes a guess for each entry, creating a confusion matrix for output.
    pub fn test(&self, input: &Array2<f32>) -> Array2<u32> {
        let mut confusion = Array2::<u32>::zeros((2, 2));

        for i in input.rows() {
            let true_prob = Bayes::class_prob(&i, &self.true_mean_std, self.true_ratio);
            let false_prob = Bayes::class_prob(&i, &self.false_mean_std, self.false_ratio);
            let class = *i.last().unwrap() as usize;
            let guess: usize = if true_prob > false_prob { 0 } else { 1 };
            confusion[[class, guess]] += 1;
        }

        confusion
    }

    // The classifier function.
    fn N(x: f32, m: f32, s: f32) -> f32 {
        1. / (f32::sqrt(TWO_PI) * s) * f32::exp(-((x - m) * (x - m) / (2. * s * s) ))
    }
    
    // Determines the probability of a specific class.
    fn class_prob(input: &ArrayView1<f32>, mean_std: &Array1<(f32, f32)>, p_class: f64) -> f64 {
        let mut prob = f64::ln(p_class);
    
        for i in 0..input.len() {
            prob += f64::ln(Bayes::N(input[i], mean_std[i].0, mean_std[i].1) as f64);
        }
    
        prob
    }


}
