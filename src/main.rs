use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotters::prelude::*;
use rand::Rng;
use std::error::Error;

// Activation functions
fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
    let s = sigmoid(x);
    &s * &(1.0 - &s)
}

// Simple Neural Network (2-layer)
struct NeuralNet {
    w1: Array2<f64>,
    b1: Array2<f64>,
    w2: Array2<f64>,
    b2: Array2<f64>,
}

impl NeuralNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        NeuralNet {
            w1: Array2::random((input_size, hidden_size), Uniform::new(-1.0, 1.0)),
            b1: Array2::zeros((1, hidden_size)),
            w2: Array2::random((hidden_size, output_size), Uniform::new(-1.0, 1.0)),
            b2: Array2::zeros((1, output_size)),
        }
    }

    fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = sigmoid(&z1);
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = sigmoid(&z2);
        (z1, a1, a2)
    }

    fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: usize, lr: f64) -> Vec<f64> {
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            // Forward
            let (z1, a1, a2) = self.forward(x);

            // Loss (MSE)
            let loss = ((&a2 - y) * (&a2 - y)).mean().unwrap();
            losses.push(loss);

            // Backpropagation
            let dz2 = (&a2 - y) * sigmoid_derivative(&a2);
            let dw2 = a1.t().dot(&dz2);
            let db2 = dz2.sum_axis(Axis(0)).insert_axis(Axis(0));

            let dz1 = dz2.dot(&self.w2.t()) * sigmoid_derivative(&z1);
            let dw1 = x.t().dot(&dz1);
            let db1 = dz1.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Update weights
            self.w1 = &self.w1 - lr * dw1;
            self.b1 = &self.b1 - lr * db1;
            self.w2 = &self.w2 - lr * dw2;
            self.b2 = &self.b2 - lr * db2;

            if epoch % 1000 == 0 {
                println!("Epoch {} | Loss: {:.6}", epoch, loss);
            }
        }
        losses
    }
}

// Vẽ loss curve
fn plot_loss(losses: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("loss_curve.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss Curve", ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..losses.len(), 0f64..1.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(x, y)| (x, *y)),
            &RED,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Đã lưu biểu đồ loss vào file: loss_curve.png");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Dữ liệu XOR
    let x = Array2::from_shape_vec(
        (4, 2),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )?;

    let y = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0])?;

    println!("Training Neural Network trên bài toán XOR...");

    let mut nn = NeuralNet::new(2, 4, 1); // 2 input -> 4 hidden -> 1 output
    let losses = nn.train(&x, &y, 10_000, 0.5);

    // Test sau khi train
    let (_, _, pred) = nn.forward(&x);
    println!("\nKết quả dự đoán sau training:");
    for (i, p) in pred.iter().enumerate() {
        println!("Input: {:?} -> Predict: {:.4} (Target: {})", 
                 x.row(i).to_vec(), p, y[[i, 0]]);
    }

    // Vẽ biểu đồ
    plot_loss(&losses)?;

    println!("\nHoàn thành! Kiểm tra file loss_curve.png");
    Ok(())
}