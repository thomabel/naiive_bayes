use crate::utility::print_data;
use crate::{Vector, Matrix, Target, Confusion};
use crate::{Input, MeanStd2};
use ndarray::prelude::*;

pub struct Bayes {
    class_prob: Vector,
    mean_stddev: MeanStd2,
}
impl Bayes {
    pub fn train(input: &Input, classes: &[&str]) -> Bayes {
        // Split the data by making an index array for each class.
        let class_len = classes.len();
        let class_index = Bayes::class_index(&input.1, classes);

        //_print_vector(&class_index.view(), "CLASS index");

        // Calculate the probability of a class given by: number of class / total number of inputs.
        let total = input.1.len_of(Axis(0)) as f32;
        let mut subtract = 0;
        let class_prob 
            = Array1::<f32>::from_shape_fn(class_len,
                |x| -> f32 {
                    let index = (class_index[x] - subtract) as f32;
                    subtract = class_index[x];
                    index / total
                }
            );

        //_print_vector(&class_prob.view(), "CLASS PROB");

        // Calculate the means and standard deviations for each class and each input in that class.
        let mean_stddev = Bayes::mean_std(&input.0, &class_index);
        //_print_matrix(&mean_stddev.view(), "MEAN STDDEV");

        Bayes { class_prob, mean_stddev }
    }

    // Determines the indicies at which the array changes class.
    fn class_index(input: &Target, classes: &[&str]) -> Array1<usize> {
        let mut lens = Array1::<usize>::zeros(classes.len());
        let mut index = 0;
        let mut current_class = classes[index];

        for (i, class) in input.iter().enumerate() {
            //print!("{}, ", class);
            if class != current_class {
                lens[index] = i;
                index += 1;
                current_class = classes[index];
                println!();
            }
        }
        //println!();
        // Assign last index to end of list.
        *lens.last_mut().unwrap() = input.len();
        lens
    }

    // Calculates the mean and standard deviation for each class to input combination.
    fn mean_std(input: &Matrix, class_input: &Array1<usize>) -> MeanStd2 {
        let classes = class_input.len();
        let entries = input.len_of(Axis(1));
        let min = 0.0001;
        let mut mean_std 
            = Array2::<(f32, f32)>::from_elem((classes, entries), (min, min));

        // For each Class
        let mut start = 0;
        for class in 0..classes {
            // Update slice bounds.
            if class > 0 {
                start = class_input[class - 1];
            }

            // For each column, create slice and iterate over it.
            let slice = input.slice(s!(start..class_input[class], ..));
            for entry in 0..entries {
                let col = slice.column(entry);
                let mean = col.mean().unwrap();
                let mut std = col.std(0.);
                if std == 0. {
                    std = min;
                }
                mean_std[[class, entry]] = (mean, std);

                print!("({}, {}), ", mean, std);
            }
            println!();
        }

        mean_std
    }


    // TEST
    pub fn test(&self, input: &Input, classes: &[&str]) -> Confusion {
        let class_len = classes.len();
        let mut confusion = Array2::<u32>::zeros((class_len, class_len));
        let mut prob = Array1::<f32>::zeros(class_len);
        let len = input.1.len();

        // For each input in the set
        for i in 0..len {
            // Calculate the probability for each class based on the input.
            //print!("{:>5}: ", i);
            let row = input.0.row(i);
            for j in 0..class_len {
                let mean_std = self.mean_stddev.row(j);
                let class_prob = self.class_prob[j];
                prob[j] = Bayes::class_probability(&row, &mean_std, class_prob);
                //print!("{}, ", prob[j]);
            }
            //println!("\n");
            
            // Find the highest probability and update the confusion matrix.
            let target = &input.1[i];
            let class = Bayes::index_from_class(classes, target).unwrap();
            let predict = Bayes::highest_prob(&prob);
            confusion[[class, predict]] += 1;

            // Reset probability array
            for k in &mut prob {
                *k = 0.;
            }
        }

        confusion
    }

    // Determine which probability was the highest.
    fn highest_prob(prob: &Vector) -> usize {
        let mut predict = 0;
        for p in prob.iter().enumerate() {
            if *p.1 > prob[predict] {
                predict = p.0;
            }
        }
        predict
    }

    // Gets the index value from a class.
    fn index_from_class(classes: &[&str], class: &str) -> Result<usize, String> {
        for i in classes.iter().enumerate() {
            if class == *i.1 {
                return Ok(i.0);
            }
        }
        Err("Class does not exist.".to_string())
    }

    // Get the probability of an input for a single class.
    fn class_probability(input: &ArrayView1<f32>, mean_std: &ArrayView1<(f32, f32)>, class_prob: f32) -> f32 {
        let mut prob = f32::ln(class_prob);

        for i in 0..input.len() - 1 {
            let n = f32::ln(Bayes::classify(input[i], mean_std[i].0, mean_std[i].1));
            prob += n;
            //print!("{}, ", n);
        }
        //println!{"\n"};

        prob
    }

    // The classifier function.
    fn classify(x: f32, m: f32, s: f32) -> f32 {
        1. / (f32::sqrt(2. * std::f32::consts::PI) * s) * 
        f32::exp(-((x - m) * (x - m) / (2. * s * s)))
    }

}
