import { useEffect, useState, FC } from "react";
import { listen } from '@tauri-apps/api/event';

const BackendLog: FC = () => {
  const [backendMsgs, setBackendMsgs] = useState<string[]>([])

  useEffect(() => {
    let unlisten = listen<string>("backend_message", (msg) => {
      console.log(msg);
      setBackendMsgs((prev) => [...prev, msg.payload])
    })

    return () => {
      unlisten.then((f) => f())
    }
  })

  return (
    <div>
      {
        backendMsgs.map((msg, idx) => <p key={idx}>{msg}</p>)
      }
    </div>
  );
}

export default BackendLog;