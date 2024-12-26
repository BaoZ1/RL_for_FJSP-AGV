/** @jsxImportSource @emotion/react */

import { useState } from "react";
import BackendMonitor from "./BackendMonitor";
import EnvEditor from "./EnvEditor";
import EnvViewer from "./EnvViewer";
import { css } from '@emotion/react'
import { Drawer, FloatButton } from "antd"
import { ApiOutlined } from '@ant-design/icons';
import { EnvState } from "./types";

function App() {
  const [showBackendDrawer, setShowBackendDrawer] = useState(false)
  const [envState, setEnvState] = useState<EnvState | null>(null)
  const [isPlanning, setIsPlanning] = useState<boolean>(false)

  const handelStart = () => {

    setIsPlanning(true)
  }

  const handelReture = () => {

    setIsPlanning(false)
  }

  return (
    <div css={css`
      width: 100%;
      height: 100%;
    `}>
      {
        isPlanning ?
        <EnvViewer state={envState!} onReture={handelReture} css={css`
          width: 100%;
          height: 100%;
          border: 10px solid transparent;
        `} />
        :
        <EnvEditor state={envState} setState={setEnvState} onStart={handelStart} css={css`
          width: 100%;
          height: 100%;
          border: 10px solid transparent;
        `} />
      }
      
      <FloatButton onClick={() => setShowBackendDrawer(true)} icon={<ApiOutlined />}/>
      <Drawer placement={"bottom"} open={showBackendDrawer} closable={false} onClose={()=>setShowBackendDrawer(false)}>
        <BackendMonitor />
      </Drawer>
    </div>
  )
}

export default App
