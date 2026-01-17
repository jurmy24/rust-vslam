use anyhow::Result;

fn main() -> Result<()> {
    println!("Run the visual-inertial demo with:");
    println!("  cargo run --bin vislam_euroc -- <path_to_euroc_mav0>");
    Ok(())
}
