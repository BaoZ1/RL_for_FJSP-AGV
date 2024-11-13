/** @jsxImportSource @emotion/react */

import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import BackendMonitor from "./BackendMonitor";
import EnvEditor from "./EnvEditor";
import { css } from '@emotion/react'
import { Drawer, Button, FloatButton } from "antd"
import { ApiOutlined } from '@ant-design/icons';

function App() {
  const [showBackendDrawer, setShowBackendDrawer] = useState(false)

  useEffect(() => {
    const p = invoke("launch_backend")
    return () => {
    //   p.then(() => invoke("close_backend"))
    }
  }, [])


  return (
    <div css={css`
        width: 100%;
        height: 100%;
      `}>
      <EnvEditor css={css`
        width: 100%;
        height: 100%;
        border: 10px solid transparent;
      `} />
      <FloatButton onClick={() => setShowBackendDrawer(true)} icon={<ApiOutlined />}/>
      <Drawer placement={"bottom"} open={showBackendDrawer} closable={false} onClose={()=>setShowBackendDrawer(false)}>
        <BackendMonitor />
      </Drawer>
    </div>
  )
}

export default App
