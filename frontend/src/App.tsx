import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import BackendLog from "./BackendLog";

function App() {
  const [backendLaunched, setBackendLaunched] = useState(false);

  function launchBackend() {
    invoke("launch_backend", { port: 2334 }).then(() => setBackendLaunched(true));
  }

  function closeBackend() {
      invoke("close_backend").then(() => setBackendLaunched(false));
  }

  return (
    <div>
      {
        backendLaunched
          ?
          <>
            <button onClick={closeBackend}>close</button>
            <BackendLog />
          </>
          : <button onClick={launchBackend}>launch</button>
      }
    </div>
  );
}

export default App;
