use std::{sync::Mutex, thread::sleep, time::Duration};
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use sysinfo::{System, Pid};

struct AppState {
    backend_pids: Vec<Pid>,
}

impl AppState {
    fn new() -> Self {
        AppState { backend_pids: Vec::new() }
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn start_backend(app: tauri::AppHandle, _port: u32) -> Result<(), String> {
    let state = app.state::<Mutex<AppState>>();
    let mut state_data = state.lock().unwrap();

    if state_data.backend_pids.len() > 0 {
        return Ok(())
    }

    let (_, child) = app
        .shell()
        .sidecar("backend")
        .unwrap()
        // .args(["-p", &port.to_string()])
        .spawn()
        .expect("Failed to spawn");

    sleep(Duration::from_millis(500));

    let mut sys = System::new_all();
    sys.refresh_all();

    if let Some(process) = sys.process(Pid::from_u32(child.pid())) {
        let name = process.name();
        println!("backend name: {}", name.to_str().unwrap());
        for (pid, process) in sys.processes()
        {
            if process.name() == name {
                state_data.backend_pids.push(*pid);
                println!("found backend process: {}", pid.to_string());
            }
        }
        return Ok(())
    }
    else {
        return Err("fail to track backend".into())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .manage::<Mutex<AppState>>(Mutex::new(AppState::new()))
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::Destroyed => {
                let state = window.state::<Mutex<AppState>>();
                let state_data = state.lock().unwrap();

                let mut sys = System::new_all();
                sys.refresh_all();
                for pid in state_data.backend_pids.iter() {
                    if let Some(process) = sys.process(*pid) {
                        if !process.kill() {
                            panic!("fail to kill backend process")
                        }
                    }
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![greet, start_backend])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
