/** @jsxImportSource @emotion/react */

import { useEffect, useState, FC } from "react";
import { listen } from '@tauri-apps/api/event';
import { Button, Flex } from "antd"
import { invoke } from "@tauri-apps/api/core";
import { css } from "@emotion/react";

enum MsgType { message, error }

const BackendMonitor: FC<{ className?: string }> = ({ className }) => {
  const [backendMsgs, setBackendMsgs] = useState<{ type: MsgType, content: string }[]>([])
  const [restarting, setRestarting] = useState<boolean>(false)

  const restart = async () => {
    setRestarting(true)
    await invoke("close_backend")
    await invoke("launch_backend")
    setRestarting(false)
  }

  useEffect(() => {
    let unlisten = listen<string>("backend_message", (msg) => {
      console.log(msg.payload)
      setBackendMsgs((prev) => [...prev, {type: MsgType.message, content: msg.payload}])
    })

    return () => {
      unlisten.then((f) => f())
    }
  }, [])

  useEffect(() => {
    let unlisten = listen<string>("backend_error", (err) => {
      console.log(err.payload)
      setBackendMsgs((prev) => [...prev, { type: MsgType.error, content: err.payload }])
    })

    return () => {
      unlisten.then((f) => f())
    }
  }, [])

  return (
    <div className={className} css={css`
      height: 100%;
    `}>
      <Flex gap="small">
        <Button color="default" variant="solid" onClick={restart} loading={restarting}>启动 / 重启</Button>
        <Button color="default" variant="filled" onClick={()=>setBackendMsgs([])}>清空</Button>
      </Flex>
      <div css={css`
        height: 100%;
        border: 1px solid gray;
        border-radius: 10px;
        padding: 5px;
        max-height: 290px; 
        overflow: auto;
        margin-top: 10px;
      `}>
        {backendMsgs.map(({content}, idx) => (
          <div key={idx} >
            {content}
          </div>
        ))}
      </div>
    </div>
  );
}

export default BackendMonitor