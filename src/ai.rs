use chrono::Local;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AIModel {
    name: String,
    provider: String,
    parameters: f64,        // Tỷ parameter (B)
    context_window: u32,    // tokens
    speed: f64,             // tokens/giây (ước tính)
    strength: String,
    weakness: String,
    cost: f64,              // $/1M tokens (input)
    release_year: u16,
}

impl AIModel {
    fn composite_score(&self) -> f64 {
        let param_score = (self.parameters / 1000.0).min(1.0); // max ~1T params
        let context_score = (self.context_window as f64 / 1_000_000.0).min(1.0);
        let speed_score = (self.speed / 200.0).min(1.0);
        let cost_score = (1.0 / (self.cost + 0.1)).min(1.0);

        (param_score * 0.3 + context_score * 0.25 + speed_score * 0.25 + cost_score * 0.2) * 100.0
    }
}

fn get_current_models() -> Vec<AIModel> {
    vec![
        AIModel {
            name: "GPT-5.5".to_string(),
            provider: "OpenAI".to_string(),
            parameters: 1800.0,
            context_window: 200_000,
            speed: 85.0,
            strength: "Reasoning, coding, multimodal".to_string(),
            weakness: "Chi phí cao".to_string(),
            cost: 15.0,
            release_year: 2026,
        },
        AIModel {
            name: "Claude 4 Opus".to_string(),
            provider: "Anthropic".to_string(),
            parameters: 2000.0,
            context_window: 500_000,
            speed: 65.0,
            strength: "Long context, safety, writing".to_string(),
            weakness: "Chậm hơn".to_string(),
            cost: 18.0,
            release_year: 2026,
        },
        AIModel {
            name: "Gemini 2.5 Pro".to_string(),
            provider: "Google".to_string(),
            parameters: 1500.0,
            context_window: 1_000_000,
            speed: 110.0,
            strength: "Multimodal, search integration".to_string(),
            weakness: "Censorship".to_string(),
            cost: 7.0,
            release_year: 2026,
        },
        AIModel {
            name: "Llama 4 Maverick".to_string(),
            provider: "Meta".to_string(),
            parameters: 405.0,
            context_window: 1_000_000,
            speed: 140.0,
            strength: "Open source, cost effective".to_string(),
            weakness: "Yếu reasoning phức tạp".to_string(),
            cost: 0.0,
            release_year: 2026,
        },
        AIModel {
            name: "Grok 4".to_string(),
            provider: "xAI".to_string(),
            parameters: 1200.0,
            context_window: 256_000,
            speed: 95.0,
            strength: "Real-time knowledge, humor".to_string(),
            weakness: "Ít multimodal".to_string(),
            cost: 10.0,
            release_year: 2026,
        },
        AIModel {
            name: "DeepSeek R1".to_string(),
            provider: "DeepSeek".to_string(),
            parameters: 671.0,
            context_window: 128_000,
            speed: 160.0,
            strength: "Coding, math, open weights".to_string(),
            weakness: "English bias".to_string(),
            cost: 0.5,
            release_year: 2026,
        },
    ]
}

fn print_analysis(models: &[AIModel]) {
    println!("=== PHÂN TÍCH CÁC MÔ HÌNH AI LỚN NHẤT THẾ GIỚI {} ===\n", 
             Local::now().format("%d/%m/%Y"));
    
    for model in models {
        println!("📌 {}", model.name);
        println!("   Provider     : {}", model.provider);
        println!("   Parameters   : {:.1}B", model.parameters);
        println!("   Context      : {}K tokens", model.context_window / 1000);
        println!("   Speed        : {:.1} tokens/s", model.speed);
        println!("   Cost         : ${:.2}/1M tokens", model.cost);
        println!("   Strengths    : {}", model.strength);
        println!("   Weaknesses   : {}", model.weakness);
        println!("   Score        : {:.1}/100\n", model.composite_score());
    }
}

fn plot_comparison(models: &[AIModel]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("ai_models_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let scores: Vec<f64> = models.iter().map(|m| m.composite_score()).collect();
    let params: Vec<f64> = models.iter().map(|m| m.parameters).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("So sánh điểm số các mô hình AI 2026", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0..models.len(), 0f64..110.0)?;

    chart.configure_mesh()
        .x_labels(0)
        .x_label_formatter(&|i| names[*i as usize].clone())
        .draw()?;

    // Bar chart score
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(BLUE.mix(0.7).filled())
            .data(scores.iter().enumerate().map(|(i, &s)| (i, s))),
    )?;

    root.present()?;
    println!("✅ Đã lưu biểu đồ so sánh vào: ai_models_comparison.png");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let models = get_current_models();
    
    print_analysis(&models);
    plot_comparison(&models)?;

    println!("\n💡 Gợi ý:");
    println!("   - Best overall      : Gemini 2.5 Pro (cân bằng)");
    println!("   - Best open source  : Llama 4 Maverick");
    println!("   - Best coding       : DeepSeek R1");
    println!("   - Best long context : Claude 4 Opus");

    Ok(())
}