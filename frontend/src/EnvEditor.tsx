/** @jsxImportSource @emotion/react */

import { FC, useEffect, useState, useRef, useMemo } from "react"
import { Splitter, Button, Layout, Card, Empty, Flex, Modal, Form } from "antd"
import { fetch } from '@tauri-apps/plugin-http';
import { open } from "@tauri-apps/plugin-dialog"
import { css } from "@emotion/react";
import { useFloating, useClientPoint, useInteractions, useHover } from '@floating-ui/react';

type EnvState = {
  inited: boolean
  timestamp: number
  operations: {
    id: number
    status: number
    machine_type: number
    processing_machine: number | null
    finish_timestamp: number
    predecessors: number[]
    arrived_preds: number[]
    successors: number[]
    sent_succs: number[]
  }[]
  machines: {
    id: number
    type: number
    status: number
    working_operation: number | null
    waiting_operation: number | null
    materials: { from: number, to: number }[]
    products: { from: number, to: number }[]
  }[]
  AGVs: {
    id: number
    status: number
    speed: number
    position: number
    target_machine: number
    loaded_item: { from: number, to: number } | null
    target_item: { from: number, to: number } | null
    finish_timestamp: number
  }[]
  distances: Record<number, Record<number, number>>
  next_operation_id: number
  next_machine_id: number
  next_AGV_id: number
}

type GenerationParams = {
  operation_count: number
  machine_count: number
  AGV_count: number
  machine_type_count: number
  min_transport_time: number
  max_transport_time: number
  min_max_speed_ratio: number
  min_process_time: number
  max_process_time: number
}

const OperationNode: FC<{ className?: string, operationId: number }> = ({ className, operationId }) => {
  return (
    <div className={className}>
      {operationId}
    </div>
  )
}

const OperationLine: FC<{ className?: string, pid: number, sid: number }> = ({ className, pid, sid }) => {
  const [isOpen, setIsOpen] = useState(false);

  const { refs, floatingStyles, context } = useFloating({
    placement: "top",
    open: isOpen,
    onOpenChange: setIsOpen,
  });
  const clientPoint = useClientPoint(context);
  const hover = useHover(context);
  const { getReferenceProps, getFloatingProps } = useInteractions([clientPoint, hover]);
  return (
    <>
      <div className={className} ref={refs.setReference} {...getReferenceProps()} />
      {isOpen && (
        <div
          ref={refs.setFloating}
          style={floatingStyles}
          {...getFloatingProps()}
        >
          {`${pid}->${sid}`}
        </div>
      )}
    </>
  )
}

const EnvEditor: FC<{ className?: string }> = ({ className }) => {
  const [envState, setEnvState] = useState<EnvState | null>(null)
  const [operationPosList, setOperationPosList] = useState<{ id: number, x: number, y: number }[]>([])
  const [graphOffset, setGraphOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })
  const [isModalOpen, setIsModalOpen] = useState(false)
  const paramsForm = Form.useForm()

  const loadEnv = async () => {
    const path = await open()
    if (path === null) {
      return
    }
    const response = await fetch(`http://localhost:8000/local_save?${new URLSearchParams({ path })}`, { method: "GET" })
    const data = await response.json() as EnvState
    console.log(data);

    setEnvState(data)
  }

  const newEnv = async () => {
    const response = await fetch(`http://localhost:8000/new_env`, { method: "GET" })
    const data = await response.json() as EnvState

    setEnvState(data)
  }

  const randEnv = async () => {
    const response = await fetch(`http://localhost:8000/local_save`, { method: "GET" })
    const data = await response.json() as EnvState

    setEnvState(data)
  }

  useEffect(() => {
    if (envState === null) {
      setOperationPosList([])
    }
    else {
      let predecessors_info = envState.operations.map((item) => ({ id: item.id, remains: item.predecessors }))
      const layers: number[][] = []
      while (predecessors_info.length !== 0) {
        const new_layer: number[] = []
        for (const { id, remains } of predecessors_info) {
          if (remains.length === 0) {
            new_layer.push(id)
          }
        }
        if (new_layer.length !== 0) {
          predecessors_info = predecessors_info.filter((item) => new_layer.find((v) => v === item.id) === undefined)
          for (const id of new_layer) {
            predecessors_info.forEach((item) => item.remains = item.remains.filter((v) => v !== id))
          }
          layers.push(new_layer)
        }
      }
      const pos_list: { id: number, x: number, y: number }[] = []
      for (const [col_idx, col] of layers.entries()) {
        for (const [idx, id] of col.entries()) {
          pos_list.push({ id, x: (col_idx - layers.length / 2 + 0.5) * 150, y: (idx - col.length / 2 + 0.5) * 100 })
        }
      }
      setOperationPosList(pos_list)
    }
  }, [envState])

  const graph = useMemo(() => {
    return [
      operationPosList.map(({ id, x, y }) => (
        envState!.operations.find((item) => item.id === id)!.predecessors.map((pred_id) => (
          operationPosList.find((item) => item.id === pred_id)!
        )).map(({ id: pid, x: px, y: py }) => {
          const len = Math.hypot(x - px, y - py)
          const rot = Math.asin((y - py) / len)

          return (
            <OperationLine key={`${pid}-${id}`} pid={pid} sid={id} css={css`
              width: ${len}px;
              border: 1px ${pid === 0 || id === 9999 ? "dashed" : "solid"} black;
              position: absolute;
              transform-origin: 0% 50%;
              rotate: ${rot}rad;
              left: calc(50% + ${px}px);
              top: calc(50% + ${py}px);
            `} />
          )
        })
      )),
      operationPosList.map(({ id, x, y }) => (
        <OperationNode key={id} operationId={id} css={css`
          width: 50px;
          height: 50px;
          border-radius: 25px;
          border: 2px solid gray;
          transform: translate(-50%, -50%);
          position: absolute;
          left: calc(50% + ${x}px);
          top: calc(50% + ${y}px);
          background-color: wheat;
        `} />
      ))
    ]
  }, [operationPosList, graphOffset])

  return (
    <>
      <Layout className={className}>
        <Layout.Header css={css`
          padding: 0;
          background-color: transparent;
          line-height: unset;
          height: unset;
        `}>
          <Card css={css`
            .ant-card-body {
              height: 100%;
              padding: 10px;
            }
          `}>
            <Flex justify="flex-start" align="center" gap="small">
              <Button type="primary" onClick={loadEnv}>读取</Button>
              <Button type="primary" onClick={newEnv}>新建</Button>
              <Button type="primary" onClick={()=>setIsModalOpen(true)}>随机</Button>
            </Flex>
          </Card>
        </Layout.Header>
        <Layout.Content css={css`
        margin-top: 5px;
      `}>
          <Card css={css`
            height: 100%;

            .ant-card-body {
              height: 100%;
              padding: 0;
            }
          `}>
            <Splitter>
              <Splitter.Panel defaultSize="70%">
                <div css={css`
                  width: 100%;
                  height: 100%;
                  position: relative;
                  overflow: hidden;
                `}>
                  {
                    envState === null
                      ?
                      <Flex justify="center" align="center" css={css`
                      height: 100%;
                    `}>
                        <Empty>
                          <Button type="primary" onClick={newEnv}>新建</Button>
                        </Empty>
                      </Flex>
                      :
                      graph
                  }
                </div>
              </Splitter.Panel>
              <Splitter.Panel>
                {
                  envState === null
                    ?
                    <Flex justify="center" align="center" css={css`
                    height: 100%;
                  `}>
                      <Empty>
                        <Button type="primary" onClick={newEnv}>新建</Button>
                      </Empty>
                    </Flex>
                    :
                    <></>
                }
              </Splitter.Panel>
            </Splitter>
          </Card>
        </Layout.Content>
      </Layout>
      <Modal title="参数设置" open={isModalOpen} footer={null} onCancel={()=>setIsModalOpen(false)}>
        <Form>
          
        </Form>
      </Modal>
    </>
  )
}

export default EnvEditor