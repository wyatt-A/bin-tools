use std::{collections::HashMap, error::Error, iter::Sum, mem::zeroed, ops::{Add, AddAssign, Deref}, time::Instant};
use rand::{thread_rng, Rng};
use num_traits::{real::Real, Num, Zero};


fn main() {
    println!("Hello, world!");


    // traverse set of coordinates,
    // recording min and max as we go ...

    let mut rng = thread_rng();

    

    let n = 30_000;
    let x:Vec<f32> = (0..n).map(|_| rng.gen_range(-1000. .. 1000.) ).collect();
    let y:Vec<f32> = (0..n).map(|_| rng.gen_range(-1000. .. 1000.) ).collect();
    let vals:Vec<f32> = (0..n).map(|_| rng.gen_range(1. .. 2.) ).collect();


    let m = mean(&x);


    let n_elems = x.len();

    let bin = Bin2D::new(&x,&y,&vals,1.);


}



pub struct Bin2D<T> {
    storage:HashMap<(i32,i32),Vec<T>>,
    min:(i32,i32),
    max:(i32,i32)
}

impl<T> Bin2D<T> where T:Copy {
    pub fn new<G>(x:&[G],y:&[G],vals:&[T],bin_size:f32) -> Self
    where G:Into<f32> + Copy
     {

        let mut hm = HashMap::<(i32,i32),Vec<T>>::with_capacity(vals.len());
    
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
    
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        
        for (v,(x,y)) in vals.into_iter().zip(x.into_iter().zip(y.into_iter())) {
            let x_coord = bin((*x).into(),bin_size);
            let y_coord = bin((*y).into(),bin_size);
    
            if y_coord < min_y {
                min_y = y_coord;
            }
    
            if y_coord > max_y {
                max_y = y_coord;
            }
    
            if x_coord < min_x {
                min_x = x_coord;
            }
    
            if x_coord > max_x {
                max_x = x_coord;
            }
    
            if let Some(collection) = hm.get_mut(&(x_coord,y_coord)) {
                collection.push(*v);
            }else {
                hm.insert((x_coord,y_coord), vec![*v]);
            }
        }

        Self {
            storage: hm,
            min: (min_x,min_y),
            max: (max_x,max_y),
        }
    }

}



fn mean<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> {

    if x.is_empty() {
        return None
    }

    let mut sum = T::zero();
    for t in x {
        sum = sum + *t;
    }
    sum = sum / x.len().into();

    Some(sum)
}

fn variance<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> { 

    if x.len() <= 1 {
        return None
    }

    let mean = mean(x).unwrap();

    let mut std = T::zero();
    for t in x {
        let d = *t - mean;
        std = std + d*d;
    }

    std = std / (x.len() - 1).into();

    Some(std)
}

fn standard_dev<T:Zero + Copy + Num + From<usize> + Real>(x:&[T]) -> Option<T> { 
    Some(variance(x)?.sqrt())
}

fn skewness<T:Zero + Copy + Num + From<usize> + Real>(x:&[T]) -> Option<T> {
    let mean = mean(x)?;
    let std = standard_dev(x)?;
    let mut skew = T::zero();
    for t in x {
        let d = *t - mean;
        skew = skew + d.powi(3);
    }
    skew = skew / <T as From<usize>>::from(x.len()-1) * std.powi(3);
    Some(skew)
}

fn kurtosis<T:Zero + Copy + Num + From<usize> + Real>(x:&[T]) -> Option<T> {
    let mean = mean(x)?;
    let std = standard_dev(x)?;
    let mut kurt = T::zero();
    for t in x {
        let d = *t - mean;
        kurt = kurt + d.powi(4);
    }
    kurt = kurt / <T as From<usize>>::from(x.len()-1) * std.powi(4);
    Some(kurt)
}

pub enum Statistic {
    Mean,
    Variance,
    Skewness,
    Kurtosis,
}

fn bin(x:f32,bin_size:f32) -> i32 {
    (x / bin_size) as i32
}


fn coord_to_sub(x:f64,y:f64,i_len:f64,j_len:f64) -> (usize,usize) {

    todo!()

}



// Function to convert matrix subscripts to a linear index in column-major order
fn subscripts_to_index_col_major(row: usize, col: usize, num_rows: usize, num_cols: usize) -> Result<usize,Box<dyn Error>> {
    if !(row < num_rows && col < num_cols) {
        Err("invalid subscripts")?
    }
    // Calculate the linear index
    Ok(col * num_rows + row)
}

// Function to convert matrix subscripts to a linear index in row-major order
fn subscripts_to_index_row_major(row: usize, col: usize, num_rows: usize, num_cols: usize) -> Result<usize,Box<dyn Error>> {
    if !(row < num_rows && col < num_cols) {
        Err("invalid subscripts")?
    }
    // Calculate the linear index
    Ok(row * num_cols + col)
}

// Function to convert a linear index to matrix subscripts in column-major order
fn index_to_subscripts_col_major(index: usize, num_rows: usize, num_cols: usize) -> Result<(usize, usize),Box<dyn Error>> {

    if !(index < num_rows * num_cols) {
        Err("invalid index")?
    }

    let col = index / num_rows;
    let row = index % num_rows;
    Ok((row, col))
}

// Function to convert a linear index to matrix subscripts in row-major order
fn index_to_subscripts_row_major(index: usize, num_rows: usize, num_cols: usize) -> Result<(usize, usize),Box<dyn Error>> {

    if !(index < num_rows * num_cols) {
        Err("invalid index")?
    }

    let row = index / num_cols;
    let col = index % num_cols;
    Ok((row, col))
}