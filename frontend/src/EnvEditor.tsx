/** @jsxImportSource @emotion/react */

import { FC, useEffect, useState, useMemo, MouseEvent, WheelEvent, useRef } from "react"
import { Splitter, Button, Layout, Card, Empty, Flex, Modal, Form, InputNumber, Space, FormInstance, FloatButton } from "antd"
import { open } from "@tauri-apps/plugin-dialog"
import { css } from "@emotion/react";
import { useFloating, useClientPoint, useInteractions, useHover, offset, safePolygon, arrow } from '@floating-ui/react';
import { RedoOutlined, CaretRightFilled, AimOutlined, PlusCircleOutlined, CloseCircleOutlined, PlusOutlined } from '@ant-design/icons';
import { OperationState, EnvState, GenerationParams, AGVState } from "./types";
import { loadEnv, newEnv, randEnv } from "./backend-api";

const OperationNode: FC<{
  className?: string,
  operation: { id: number, x: number, y: number },
  radius: number,
  scaleRate: number
}> = (props) => {
  const type = ({ 0: "start", 9999: "end" } as const)[props.operation.id] || "normal"

  const scaleRateRef = useRef(props.scaleRate)
  useEffect(() => { scaleRateRef.current = props.scaleRate }, [props.scaleRate])

  const [reference, setReference] = useState<HTMLElement | null>(null)

  const [isAddSuccOpen, setIsAddSuccOpen] = useState(false);
  const {
    refs: addSuccRefs,
    floatingStyles: addSuccFloatingStyles,
    context: addSuccContext
  } = useFloating({
    placement: "right",
    open: isAddSuccOpen,
    onOpenChange: setIsAddSuccOpen,
    elements: { reference },
    middleware: [offset(({ x, y, rects: { floating: { height } } }) => {
      const placement_offset_x = 0
      const placement_offset_y = -height / 2
      const raw_offset_x = x - placement_offset_x
      const raw_offset_y = y - placement_offset_y
      const scaled_offset_offset = 5 / scaleRateRef.current

      return {
        mainAxis: -raw_offset_x + raw_offset_x / scaleRateRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_y + raw_offset_y / scaleRateRef.current
      }
    })]
  });
  const addSuccHover = useHover(addSuccContext, { handleClose: safePolygon() });
  const { getFloatingProps: getAddSuccFloatingProps } = useInteractions([addSuccHover]);

  const [isAddPredOpen, setIsAddPredOpen] = useState(false);
  const {
    refs: addPredRefs,
    floatingStyles: addPredFloatingStyles,
    context: addPredContext
  } = useFloating({
    placement: "left",
    open: isAddSuccOpen,
    onOpenChange: setIsAddPredOpen,
    elements: { reference },
    middleware: [offset(({ x, y, rects: { floating: { width, height } } }) => {
      const placement_offset_x = width
      const placement_offset_y = -height / 2
      const raw_offset_x = -x - placement_offset_x
      const raw_offset_y = y - placement_offset_y
      const scaled_offset_offset = 5 / scaleRateRef.current

      return {
        mainAxis: -raw_offset_x + raw_offset_x / scaleRateRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_y + raw_offset_y / scaleRateRef.current
      }
    })]
  });
  const addPredHover = useHover(addPredContext, { handleClose: safePolygon() });
  const { getFloatingProps: getAddPredFloatingProps } = useInteractions([addPredHover]);

  const [isRemoveOpen, setIsRemoveOpen] = useState(false);
  const {
    refs: removeRefs,
    floatingStyles: removeFloatingStyles,
    context: removeContext
  } = useFloating({
    placement: "bottom",
    open: isAddSuccOpen,
    onOpenChange: setIsRemoveOpen,
    elements: { reference },
    middleware: [offset(({ x, y, rects: { floating: { width } } }) => {
      const placement_offset_x = - width / 2
      const placement_offset_y = 0
      const raw_offset_x = x - placement_offset_x
      const raw_offset_y = y - placement_offset_y
      const scaled_offset_offset = 5 / scaleRateRef.current

      return {
        mainAxis: -raw_offset_y + raw_offset_y / scaleRateRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_x + raw_offset_x / scaleRateRef.current
      }
    })]
  });
  const removeHover = useHover(removeContext, { handleClose: safePolygon() });
  const { getFloatingProps: getRemoveFloatingProps } = useInteractions([removeHover]);

  const { getReferenceProps } = useInteractions([addSuccHover, addPredHover, removeHover])

  return (
    <>
      <div className={props.className} ref={setReference} {...getReferenceProps()}
        css={css`
          width: ${props.radius * 2}px;
          height: ${props.radius * 2}px;
          border-radius: ${props.radius}px;
          border: 2px ${type === "normal" ? "solid" : "dashed"} gray;
          transform: translate(-50%, -50%);
          position: absolute;
          left: ${props.operation.x}px;
          top: ${props.operation.y}px;
          background-color: wheat;
          z-index: 1;
          :hover {
            border-color: red;
          }
        `}
      >
        {props.operation.id}
      </div>
      {
        isAddSuccOpen && type !== "end" && (
          <PlusCircleOutlined css={css`
              font-size: 25px;
              position: absolute;
              top: 50%;
              left: 50%;
              transform: translate(-50%, -50%);
              background-color: white;
              border-radius: 50%;
              color: #888888;

              :hover {
                color: black;
              }
            `}
            ref={addSuccRefs.setFloating} {...getAddSuccFloatingProps()} style={addSuccFloatingStyles}
          />
        )
      }
      {
        isAddPredOpen && type !== "start" && (
          <PlusCircleOutlined css={css`
            font-size: 25px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            border-radius: 50%;
            color: #888888;

            :hover {
              color: black;
            }
          `}
            ref={addPredRefs.setFloating} {...getAddPredFloatingProps()} style={addPredFloatingStyles}
          />
        )
      }
      {
        isRemoveOpen && type === "normal" && (
          <CloseCircleOutlined css={css`
            font-size: 25px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            border-radius: 50%;
            color: #fe9b9b;

            :hover {
              color: red;
            }
          `}
            ref={removeRefs.setFloating} {...getRemoveFloatingProps()} style={removeFloatingStyles}
          />
        )
      }
    </>
  )
}

const OperationLine: FC<{
  className?: string,
  p: { id: number, x: number, y: number },
  s: { id: number, x: number, y: number },
  nodeRadius: number,
  scaleRate: number
}> = (props) => {
  const [isOpen, setIsOpen] = useState(false);
  const scaleRateRef = useRef(props.scaleRate)
  useEffect(() => { scaleRateRef.current = props.scaleRate }, [props.scaleRate])
  const { refs, floatingStyles, context } = useFloating({
    placement: "top",
    open: isOpen,
    onOpenChange: setIsOpen,
    middleware: [offset(({ x, y, rects: { floating: { width, height } } }) => {
      const top_offset_x = -width / 2
      const top_offset_y = height
      const raw_offset_x = x - top_offset_x
      const raw_offset_y = -y - top_offset_y
      const scaled_offset_offset = 10 / scaleRateRef.current

      return {
        mainAxis: -raw_offset_y + raw_offset_y / scaleRateRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_x + raw_offset_x / scaleRateRef.current
      }
    })]
  });
  const clientPoint = useClientPoint(context);
  const hover = useHover(context);
  const { getReferenceProps, getFloatingProps } = useInteractions([clientPoint, hover]);

  const raw_len = Math.hypot(props.s.x - props.p.x, props.s.y - props.p.y)
  const len = raw_len - props.nodeRadius * 2
  const rot = Math.asin((props.s.y - props.p.y) / raw_len)

  const center_x = (props.p.x + props.s.x) / 2 - len / 2
  const center_y = (props.p.y + props.s.y) / 2


  return (
    <>
      <div className={props.className}
        ref={refs.setReference} {...getReferenceProps()} css={css`
        width: ${len}px;
        border: 1px ${props.p.id === 0 || props.s.id === 9999 ? "dashed" : "solid"} black;
        position: absolute;
        rotate: ${rot}rad;
        left: calc(50% + ${center_x}px);
        top: calc(50% + ${center_y}px);
        z-index: 0;
        :hover {
          border-color: red;
          z-index: 4;
          & > * {
            color: red;
          }
        }
      `}
      >
        <CaretRightFilled css={css`
          position: absolute;
          right: 0;
          translate: 40% -50%;
        `} />
      </div>
      {isOpen && (
        <div
          ref={refs.setFloating}
          style={floatingStyles}
          {...getFloatingProps()}
          css={css`
            width: max-content;
            background-color: white;
            padding: 3px;
            border: 1px solid #818181;
            border-radius: 5px;
            z-index: 5;
          `}
        >
          {`${props.p.id}->${props.s.id}`}
        </div>
      )}
    </>
  )
}

const GraphEditor: FC<{ className?: string, states: OperationState[] }> = (props) => {
  const [operationPosList, setOperationPosList] = useState<{ id: number, x: number, y: number }[]>([])
  const [graphOffset, setGraphOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })
  const [scaleRatio, setScaleRatio] = useState<number>(1)

  const colSpace = 150
  const rowSpace = 125

  const nodeRadius = 25

  const resetPos = () => {
    setGraphOffset({ x: 0, y: 0 })
    setScaleRatio(1)
  }

  const fix_sort = () => {
    let predecessors_info: {
      id: number,
      remains: number[],
      preds_pos: number[]
    }[] = props.states.map((item) => ({ id: item.id, remains: item.predecessors, preds_pos: [] }))

    const layers: number[][] = []

    while (predecessors_info.length !== 0) {
      const new_layer: number[] = []
      const expect_pos: Record<number, number> = {}

      for (const { id, remains, preds_pos } of predecessors_info) {
        if (remains.length === 0) {
          new_layer.push(id)
          expect_pos[id] = preds_pos.reduce((sum, n) => sum + n, 0) / preds_pos.length
        }
      }

      new_layer.sort((a, b) => expect_pos[a] - expect_pos[b])

      predecessors_info = predecessors_info.filter((item) => !new_layer.includes(item.id))

      for (const [idx, id] of new_layer.entries()) {
        predecessors_info.forEach((item) => {
          if (item.remains.includes(id)) {
            item.preds_pos.push(idx / new_layer.length)
            item.remains = item.remains.filter((rid) => rid !== id)
          }
        })
      }
      layers.push(new_layer)
    }
    const pos_list: { id: number, x: number, y: number }[] = []
    for (const [col_idx, col] of layers.entries()) {
      for (const [idx, id] of col.entries()) {
        pos_list.push({
          id,
          x: (col_idx - layers.length / 2 + 0.5) * colSpace,
          y: (idx - col.length / 2 + 0.5) * rowSpace
        })
      }
    }
    setOperationPosList(pos_list)
  }

  useEffect(fix_sort, [props.states])

  const trackGraphDrag = (e: MouseEvent) => {
    e.preventDefault()
    let prev_x = e.clientX
    let prev_y = e.clientY

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const dx = e.clientX - prev_x
      const dy = e.clientY - prev_y
      setGraphOffset(({ x, y }) => ({ x: x + dx, y: y + dy }))
      prev_x = e.clientX
      prev_y = e.clientY
    }
    const handleMouseUp = () => {
      removeEventListener('mousemove', handleMouseMove)
      removeEventListener('mouseup', handleMouseUp)
    }
    addEventListener('mousemove', handleMouseMove)
    addEventListener('mouseup', handleMouseUp)
  }

  const updateScale = (e: WheelEvent) => {
    const rate = e.deltaY < 0 ? 1.1 : (1 / 1.1)
    setScaleRatio((prev) => prev * rate)
    setGraphOffset(({ x: px, y: py }) => ({ x: px * rate, y: py * rate }))
  }

  const graph_doms = useMemo(() => [
    operationPosList.map((s) => (
      props.states.find((item) => item.id === s.id)!.predecessors.map((pred_id) => (
        operationPosList.find((item) => item.id === pred_id)!
      )).map((p) => (
        <OperationLine key={`${p.id}-${s.id}`} p={p} s={s} nodeRadius={nodeRadius} scaleRate={scaleRatio} />
      ))
    )),
    operationPosList.map((operation) => (
      <OperationNode key={operation.id} operation={operation} radius={nodeRadius} scaleRate={scaleRatio} />
    ))
  ], [operationPosList, scaleRatio])

  return (
    <div className={props.className}
      onMouseDown={trackGraphDrag} onWheel={updateScale}
      css={css`
        position: relative;
        overflow: hidden;
      `}
    >
      <div css={css`
        width: 0;
        height: 0;
        /* border: 5px solid red; */
        position: absolute;
        overflow: visible;
        top: calc(50% + ${graphOffset.y}px);
        left: calc(50% + ${graphOffset.x}px);
        scale: ${scaleRatio};
      `}>
        {
          graph_doms
        }
      </div>
      <FloatButton onClick={resetPos} icon={<AimOutlined />} css={css`
        position: absolute;
        bottom: 48px;
        right: 24px;
      `} />
    </div>
  )
}

const AGVNode: FC<{state: AGVState}> = (props) => {
  const [reference, setReference] = useState<HTMLElement | null>(null)
  const arrowRef = useRef(null)

  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const { refs: infoRefs, floatingStyles: infoFloatingStyles, context: infoContexts, middlewareData } = useFloating({
    placement: "top",
    open: isInfoOpen,
    onOpenChange: setIsInfoOpen,
    elements: {reference},
    middleware: [offset(10), arrow({element: arrowRef})]
  });
  const infoHover = useHover(infoContexts);
  const { getFloatingProps: getInfoFloatingProps } = useInteractions([infoHover]);

  const [isRemoveOpen, setIsRemoveOpen] = useState(false);
  const { refs: removeRefs, floatingStyles: removeFloatingStyles, context: removeContexts } = useFloating({
    placement: "bottom-end",
    open: isRemoveOpen,
    onOpenChange: setIsRemoveOpen,
    elements: { reference }
  });
  const removeHover = useHover(removeContexts, { handleClose: safePolygon() });
  const { getFloatingProps: getRemoveFloatingProps } = useInteractions([removeHover]);

  const { getReferenceProps } = useInteractions([infoHover, removeHover])

  return (
    <>
      <div ref={setReference} {...getReferenceProps()} css={css`
        width: 50px;
        height: 50px;
        background-color: black;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 50%;
      `}>
        <div css={css`
          color: white;
          font-size: larger;
        `}>
          {props.state.id}
        </div>
      </div>
      {
        isInfoOpen && (
          <div ref={infoRefs.setFloating} {...getInfoFloatingProps()} style={infoFloatingStyles} css={css`
            padding: 5px;
            border: 1px solid gray;
            background-color: white;
            border-radius: 5px;
          `}>
            12345679
            <div ref={arrowRef} css={css`
              position: absolute;
              left: ${middlewareData.arrow?.x}px;
              bottom: -10px;
              width: 0;
              height: 0;
              border-left: 6px solid transparent;  
              border-right: 6px solid transparent;  
              border-top: 10px solid gray; 
            `}/>
          </div>
        )
      }
      {
        isRemoveOpen && (
          <CloseCircleOutlined css={css`
            font-size: 25px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            border-radius: 50%;
            color: #fe9b9b;

            :hover {
              color: red;
            }
          `}
            ref={removeRefs.setFloating} {...getRemoveFloatingProps()} style={removeFloatingStyles}
          />
        )
      }
    </>
  )
}

const RandParamConfigForm: FC<{ formData: FormInstance<GenerationParams>, onFinish: (values: GenerationParams) => void }> = (props) => {

  return (
    <Form form={props.formData} initialValues={{
      operation_count: 10,
      machine_count: 5,
      AGV_count: 5,
      machine_type_count: 3,
      min_transport_time: 8,
      max_transport_time: 15,
      min_max_speed_ratio: 0.8,
      min_process_time: 8,
      max_process_time: 15
    }}
      onFinish={props.onFinish}
    >
      <h4>工序</h4>
      <Form.Item<GenerationParams> name="operation_count" label="数量">
        <InputNumber min={5} max={20} />
      </Form.Item>
      <Form.Item<GenerationParams> label="处理时间">
        <Space>
          <Form.Item<GenerationParams> noStyle name="min_process_time">
            <InputNumber min={1} max={30} step={0.05} />
          </Form.Item>
          -
          <Form.Item<GenerationParams> noStyle name="max_process_time">
            <InputNumber min={1} max={30} step={0.05} />
          </Form.Item>
        </Space>
      </Form.Item>
      <h4>设备</h4>
      <Form.Item<GenerationParams> name="machine_count" label="数量">
        <InputNumber min={2} max={10} />
      </Form.Item>
      <Form.Item<GenerationParams> name="machine_type_count" label="种类">
        <InputNumber min={1} max={5} />
      </Form.Item>
      <h4>AGV</h4>
      <Form.Item<GenerationParams> name="AGV_count" label="数量">
        <InputNumber min={2} max={10} />
      </Form.Item>
      <Form.Item<GenerationParams> label="运输时间">
        <Space>
          <Form.Item<GenerationParams> noStyle name="min_transport_time">
            <InputNumber min={1} max={30} step={0.05} />
          </Form.Item>
          -
          <Form.Item<GenerationParams> noStyle name="max_transport_time">
            <InputNumber min={1} max={30} step={0.05} />
          </Form.Item>
        </Space>
      </Form.Item>
      <Form.Item<GenerationParams> name="min_max_speed_ratio" label="速度差异">
        <InputNumber min={0.3} max={1} step={0.05} />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">确定</Button>
      </Form.Item>
    </Form>
  )
}

const EnvEditor: FC<{ className?: string }> = ({ className }) => {
  const [envState, setEnvState] = useState<EnvState | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [paramsForm] = Form.useForm<GenerationParams>()

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
              <Button type="primary" onClick={async () => {
                const path = await open()
                if (path !== null) {
                  setEnvState(await loadEnv(path))
                }
              }}>读取</Button>
              <Button type="primary" onClick={async () => setEnvState(await newEnv())}>新建</Button>
              <Button type="primary" onClick={() => setIsModalOpen(true)}>随机</Button>
              <Button type="primary" shape="circle" onClick={async () => {
                setEnvState(await randEnv(paramsForm.getFieldsValue()))
              }} icon={<RedoOutlined />} />
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
            {
              envState === null
                ?
                <Flex justify="center" align="center" css={css`
                      height: 100%;
                    `}>
                  <Empty>
                    <Button type="primary" onClick={async () => setEnvState(await newEnv())}>新建</Button>
                  </Empty>
                </Flex>
                :
                <Splitter>
                  <Splitter.Panel defaultSize="70%">
                    <GraphEditor states={envState.operations} css={css`
                      width: 100%;
                      height: 100%;
                    `} />

                  </Splitter.Panel>
                  <Splitter.Panel min="20%" collapsible>
                    <Splitter layout="vertical">
                      <Splitter.Panel defaultSize="70%">
                        <></>
                      </Splitter.Panel>
                      <Splitter.Panel>
                        <Flex gap="middle" wrap css={css`
                          padding: 15px;
                        `}>
                          {
                            envState.AGVs.map((v)=><AGVNode state={v} />)
                          }
                          <div css={css`
                            width: 50px;
                            height: 50px;
                            background-color: white;
                            border: 2px dashed gray;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            border-radius: 50%;

                            :hover {
                              border-color: red;
                              & > * {
                                color: red;
                              }
                            }
                          `}>
                            <div css={css`
                              font-size: larger;
                            `}>
                              <PlusOutlined/>
                            </div>
                          </div>
                        </Flex>
                      </Splitter.Panel>
                    </Splitter>
                  </Splitter.Panel>
                </Splitter>
            }
          </Card>
        </Layout.Content>
      </Layout>
      <Modal title="参数设置" open={isModalOpen} footer={null} onCancel={() => setIsModalOpen(false)}>
        <RandParamConfigForm formData={paramsForm}
          onFinish={async (values) => {
            setIsModalOpen(false)
            setEnvState(await randEnv(values))
          }} />
      </Modal>
    </>
  )
}

export default EnvEditor
