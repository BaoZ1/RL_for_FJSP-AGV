use std::sync::Mutex;
use sysinfo::{Pid, System};
use tauri::async_runtime::spawn;
use tauri::{Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

struct AppState {
    backend_pids: Vec<Pid>,
}

impl AppState {
    fn new() -> Self {
        AppState {
            backend_pids: Vec::new(),
        }
    }
}

const BACKEND_NAME: &str = "FJSP-AGV_backend";

fn refresh_backend_pids(pids: &mut Vec<Pid>) {
    pids.clear();

    let mut sys = System::new_all();
    sys.refresh_all();

    for (pid, process) in sys.processes() {
        if process
            .name()
            .to_str()
            .unwrap()
            .to_string()
            .starts_with(BACKEND_NAME)
        {
            pids.push(*pid);
            println!("found backend process: {}", pid.to_string());
        }
    }
}

#[tauri::command]
fn launch_backend(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<Mutex<AppState>>();
    let mut state_data = state.lock().unwrap();

    refresh_backend_pids(&mut state_data.backend_pids);
    if state_data.backend_pids.len() > 0 {
        _close_backend(&mut state_data.backend_pids).unwrap();
    }

    let (mut rx, _) = app
        .shell()
        .sidecar(BACKEND_NAME)
        .unwrap()
        // .args(["-p", &port.to_string()])
        .spawn()
        .expect("failed to spawn backend");

    let handle = app.clone();
    spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stderr(vec) => handle
                    .emit(
                        "backend_error",
                        format!("{}", String::from_utf8_lossy(&vec)),
                    )
                    .expect("failed to emit event"),
                CommandEvent::Stdout(vec) => handle
                    .emit(
                        "backend_message",
                        format!("{}", String::from_utf8_lossy(&vec)),
                    )
                    .expect("failed to emit event"),
                _ => (),
            }
        }
    });

    Ok(())
}

fn _close_backend(pids: &mut Vec<Pid>) -> Result<(), String> {
    let mut sys = System::new_all();
    sys.refresh_all();
    for (_, process) in sys.processes() {
        if process
            .name()
            .to_str()
            .unwrap()
            .to_string()
            .starts_with(BACKEND_NAME)
        {
            if !process.kill() {
                panic!("fail to kill backend process")
            }
        }
    }
    pids.clear();
    return Ok(());
}

#[tauri::command]
fn check_backend(app: tauri::AppHandle) -> bool {
    let state = app.state::<Mutex<AppState>>();
    let mut state_data = state.lock().unwrap();

    refresh_backend_pids(&mut state_data.backend_pids);

    state_data.backend_pids.len() != 0
}

#[tauri::command]
fn close_backend(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<Mutex<AppState>>();
    let mut state_data = state.lock().unwrap();

    _close_backend(&mut state_data.backend_pids)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .manage::<Mutex<AppState>>(Mutex::new(AppState::new()))
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::Destroyed => {
                let state = window.state::<Mutex<AppState>>();
                let mut state_data = state.lock().unwrap();

                _close_backend(&mut state_data.backend_pids).unwrap();
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            launch_backend,
            close_backend,
            check_backend
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
