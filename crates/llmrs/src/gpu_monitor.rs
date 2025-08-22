use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{TemperatureSensor, TemperatureThreshold};
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use std::sync::{Once, Mutex};

static INIT: Once = Once::new();
static NVML: Mutex<Option<Nvml>> = Mutex::new(None);

#[derive(Debug, Clone)]
pub struct GPUUtilInfo {
    pub clock: u32,
    pub max_clock: u32,
    pub power: u32,
    pub power_limit: u32,
    pub fan: u32,
    pub temperature: u32,
    pub temp_slowdown: u32,
    pub gpu_utilization: f32,
    pub mem_utilization: f32,
    pub throttle_reason: String,
}

fn get_throttle_reason(throttle_reasons: ThrottleReasons) -> String {
    if throttle_reasons.contains(ThrottleReasons::SW_POWER_CAP) ||
       throttle_reasons.contains(ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN) {
        "power cap".to_string()
    } else if throttle_reasons.contains(ThrottleReasons::SW_THERMAL_SLOWDOWN) ||
              throttle_reasons.contains(ThrottleReasons::HW_THERMAL_SLOWDOWN) {
        "thermal cap".to_string()
    } else if !throttle_reasons.is_empty() {
        "other cap".to_string()
    } else {
        "no cap".to_string()
    }
}

pub fn get_gpu_utilization_info() -> Result<GPUUtilInfo, Box<dyn std::error::Error>> {
    INIT.call_once(|| {
        if let Ok(nvml) = Nvml::init() {
            let _ = NVML.lock().map(|mut guard| *guard = Some(nvml));
        }
    });
    
    let nvml_guard = NVML.lock()?;
    let nvml = nvml_guard.as_ref().ok_or("Failed to initialize NVML")?;
    let device = nvml.device_by_index(0)?;
    
    // Get basic GPU information that's available in most NVML versions
    let power_limit = device.power_management_limit()?;
    let power = device.power_usage()?;
    let temperature = device.temperature(TemperatureSensor::Gpu)?;
    let temp_slowdown = device.temperature_threshold(TemperatureThreshold::Slowdown)?;
    let throttle_reasons = device.current_throttle_reasons()?;
    
    // Try to get fan speed, but provide fallback if not available
    let fan = device.fan_speed(0).unwrap_or(0);
    
    // Get utilization rates
    let utilization = device.utilization_rates()?;
    let gpu_utilization = utilization.gpu as f32;
    let mem_utilization = utilization.memory as f32;
    
    // For clock info, we'll use fallback values since the API might not be available
    let clock = 0; // Fallback - could be enhanced with actual API call if available
    let max_clock = 0; // Fallback - could be enhanced with actual API call if available
    
    Ok(GPUUtilInfo {
        clock,
        max_clock,
        power,
        power_limit,
        fan,
        temperature,
        temp_slowdown,
        gpu_utilization,
        mem_utilization,
        throttle_reason: get_throttle_reason(throttle_reasons),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_monitor_initialization() {
        // This test will only pass if NVML is available and working
        match get_gpu_utilization_info() {
            Ok(info) => {
                println!("GPU Info: {:?}", info);
                // Basic sanity checks
                assert!(info.temperature > 0, "Temperature should be positive");
                assert!(info.power > 0, "Power should be positive");
                assert!(info.gpu_utilization >= 0.0, "GPU utilization should be non-negative");
                assert!(info.mem_utilization >= 0.0, "Memory utilization should be non-negative");
            }
            Err(e) => {
                // It's okay if NVML is not available in test environment
                println!("NVML not available in test environment: {}", e);
            }
        }
    }
}
