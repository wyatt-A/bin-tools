use std::{collections::HashMap, error::Error, fs::File, io::Read, iter::Sum, mem::zeroed, ops::{Add, AddAssign, Deref}, time::Instant};
use rand::{thread_rng, Rng};
use num_traits::{real::Real, Float, Num, Zero};
use sheet_calc::SpreadSheet2D;
use tiff::encoder::{colortype::{CMYK32Float, ColorType, Gray32Float, Gray64Float}, TiffEncoder};
use rayon::prelude::*;

fn main() -> Result<(),Box<dyn Error>> {

    let now = Instant::now();

    let mut f = File::open("/Users/Wyatt/qpath_test_data/1125ganglio_slide1_detection-measurements.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    
    println!("parsing spreadsheet ...");
    let s = SpreadSheet2D::from_string(s, "\t", 0);

    println!("extracting columns ...");
    let x = s.exract_column("Centroid X")?;
    let y = s.exract_column("Centroid Y")?;

    let stats = [Statistic::Mean];


    for (header,vals) in s.column_headers().iter().zip(s.columns_numeric()) {

        println!("binning feature ...");
        let bin = Bin2D::new(&x,&y,&vals,100.);
    
        let dims = bin.dimensions();
        let width = dims.0;
        let height = dims.1;
    
        for stat in stats {
            let fname = format!("{:?}",stat);
            println!("writing {} ...",fname);
            let agg = bin.aggregate(stat);
            let file = File::create(fname + "_" + header + ".tiff").expect("Failed to create file");
            // Create a TIFF encoder
            let mut tiff_encoder = TiffEncoder::new(file).expect("Failed to create TIFF encoder");
            // Write the image data to the TIFF file
            tiff_encoder
                .write_image::<Gray64Float>(width as u32, height as u32,&agg)
                .expect("Failed to write TIFF image");
        }
    }


    // s.column_headers().par_iter().for_each(|header|{


    // });

    let dur = now.elapsed().as_millis();
    println!("took {} ms",dur);
    Ok(())
}



pub struct Bin2D<T> {
    pub storage:HashMap<(i32,i32),Vec<T>>,
    pub min:(i32,i32),
    pub max:(i32,i32)
}

impl<T> Bin2D<T> where T:Zero + Float {
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

    pub fn dimensions(&self) -> (usize,usize) {
        (
            (self.max.0 - self.min.0 + 1) as usize,
            (self.max.1 - self.min.1 + 1) as usize
        )
    }

    pub fn aggregate(&self,t:Statistic) -> Vec<T> {
        let dims = self.dimensions();
        let n = dims.0 * dims.1;
        let mut out = vec![T::nan();n];
        for (coord,thing) in &self.storage {
            let sub_x = (coord.0 - self.min.0) as usize;
            let sub_y = (coord.1 - self.min.1) as usize;
            let idx = subscripts_to_index_col_major(sub_x,sub_y,dims.0,dims.1).unwrap();
            match &t {
                Statistic::Mean => out[idx] = mean(thing).unwrap_or(T::nan()),
                Statistic::StandardDev => out[idx] = standard_dev(thing).unwrap_or(T::nan()),
                Statistic::Variance => out[idx] = variance(thing).unwrap_or(T::nan()),
                Statistic::Skewness => out[idx] = skewness(thing).unwrap_or(T::nan()),
                Statistic::Kurtosis => out[idx] = kurtosis(thing).unwrap_or(T::nan()),
                Statistic::Count => out[idx] = T::from(thing.len()).unwrap_or(T::nan()),
                
            }
        }
        out
    }

}

#[derive(Copy,Clone,Debug)]
pub enum Statistic {
    Mean,
    Variance,
    StandardDev,
    Skewness,
    Kurtosis,
    Count,
}

fn mean<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> {

    if x.is_empty() {
        return None
    }

    let mut sum = T::zero();
    for t in x {
        sum = sum + *t;
    }

    sum = sum / T::from(x.len())?;

    Some(sum)
}

fn variance<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> { 
    central_moment(x, 2)
}

fn standard_dev<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> { 
    Some(variance(x)?.sqrt())
}

fn skewness<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> {
    Some(central_moment(x, 3)? / standard_dev(x)?.powi(3))
}

fn kurtosis<T:Zero + Copy + Num + Real>(x:&[T]) -> Option<T> {
    Some(central_moment(x, 4)? / standard_dev(x)?.powi(4))
}

fn central_moment<T:Zero + Copy + Num + Real>(x:&[T],order:i32) -> Option<T> {
    let mean = mean(x)?;
    let mut moment = T::zero();
    for sample in x {
        let resid = *sample - mean;
        moment = moment + resid.powi(order);
    }
    moment = moment /  T::from(x.len())?;
    Some(moment)
}

fn bin(x:f32,bin_size:f32) -> i32 {
    (x / bin_size) as i32
}

// Function to convert matrix subscripts to a linear index in column-major order
fn subscripts_to_index_col_major(row: usize, col: usize, num_rows: usize, num_cols: usize) -> Result<usize,Box<dyn Error>> {
    if !(row < num_rows && col < num_cols) {
        Err(format!("{},{},{},{}",row,num_rows,col,num_cols))?
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