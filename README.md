# UriSense-Gen2-MAX32690
Dual-Modality Point-of-Care Urinalysis Engine powered by Analog Devices MAX32690 (ARM Cortex-M4F, TinyML &amp; BLE 5.2).

# UriSense Gen-2 (MAX Version)
### Dual-Modality Urinalysis Diagnostic Engine on Analog Devices MAX32690

UriSense Gen-2 is a portable, clinical-grade urinalysis analyzer designed for decentralized point-of-care diagnostics. Migrating from the Gen-1 optical-only prototype, Gen-2 introduces dual-modality sensing: combining multi-channel spectral absorption with precision electrochemical impedance/conductivity sensing powered by the **Analog Devices MAX32690EVKIT**.

## Key Hardware Architectural Upgrades (Gen-1 vs. Gen-2)

| Parameter | Gen-1 (PoC Baseline) | Gen-2 (MAX Architecture) |
| :--- | :--- | :--- |
| **Compute Core** | Low-cost microcontroller | 120 MHz ARM Cortex-M4F with FPU (MAX32690) |
| **Sensing Modality** | 11-Channel Optical Reflectance | Dual-Modality: Multispectral (415–940nm) + Electrochemical Impedance |
| **Analog Frontend** | Unbuffered / noisy rail ADC | 12-bit 1MSPS SAR ADC + Onboard Ultra-Low-Drift Voltage Reference |
| **Edge Intelligence** | Static threshold lookup | Quantized TinyML Reaction Kinetic Regression & Drift Normalization |
| **Telemetry & Security** | Standard BLE broadcast | BLE 5.2 with On-Chip Hardware Cryptographic Accelerator (AES/SHA/PUF) |
| **Local Interface** | Serial terminal / external phone | Onboard 1.45" ST7735 Color TFT Display |

## Target Biomarkers

1. **Physical Markers:** Specific Gravity (SG), Osmolality, Turbidity.
2. **Biochemical Reagent Pads:** Leukocytes, Nitrite, Urobilinogen, Protein, pH, Occult Blood, Ketones, Bilirubin, Glucose.
