// THROWAWAY VERIFICATION SPIKE — TASKS.md task 2.0.
// Goal: prove wreq 5.3 + wreq-util 2.2.6 build on this Win11 box and that a
// Chrome emulation profile produces a Chrome TLS/HTTP2 fingerprint.
// Smoke targets: https://tls.peet.ws/api/all  +  a Cloudflare-fronted site.
// DELETE after the spike decision is recorded.

use wreq::Client;
use wreq_util::Emulation;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .emulation(Emulation::Chrome137)
        .build()?;

    // --- 1. Fingerprint check ---------------------------------------------
    println!("== GET https://tls.peet.ws/api/all ==");
    let resp = client.get("https://tls.peet.ws/api/all").send().await?;
    println!("status: {}", resp.status());
    let body = resp.text().await?;
    println!("{body}");

    // --- 2. Cloudflare-fronted site ---------------------------------------
    // discord.com sits behind Cloudflare; a generic Rust client typically eats
    // a 403 challenge, a real Chrome fingerprint should get real content.
    println!("\n== GET https://discord.com (Cloudflare-fronted) ==");
    let cf = client.get("https://discord.com").send().await?;
    println!("status: {}", cf.status());
    let cf_body = cf.text().await?;
    let is_challenge = cf_body.contains("Just a moment")
        || cf_body.contains("challenge-platform")
        || cf_body.contains("cf-chl");
    println!("challenge_page: {is_challenge}");
    println!("body_len: {}", cf_body.len());
    println!(
        "body_head: {}",
        &cf_body.chars().take(300).collect::<String>()
    );

    Ok(())
}
