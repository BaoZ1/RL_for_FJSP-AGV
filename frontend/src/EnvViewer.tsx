/** @jsxImportSource @emotion/react */

import { useEffect, useState, MouseEvent, WheelEvent, useRef, useLayoutEffect } from "react"
import {
  Button, Card, Flex, FloatButton, Layout, Splitter,
  Typography, Dropdown, InputNumber, Modal, Timeline, TimelineItemProps, Empty,
  Progress,
} from "antd"
import {
  BaseFC, operationStatusMapper, OperationStatus, OperationStatusIdx, Action,
  AGVState, EnvState, MachineState, OperationState, actionStatusMapper,
  Paths,
  AGVStatusMapper,
  MachineStatusMapper
} from "./types"
import { css } from "@emotion/react"
import { offset, useClientPoint, useFloating, useHover, useInteractions } from "@floating-ui/react"
import { AimOutlined, CaretRightFilled, ClockCircleOutlined, CloseOutlined, LeftOutlined, RightOutlined } from '@ant-design/icons'
import { getPaths, initEnv, loadModel, modelList, predict, removeModel } from "./backend-api"
import { open } from "@tauri-apps/plugin-dialog"

const operationColor = {
  blocked: "#5b5b5b",
  unscheduled: "#db9200",
  waiting: "#b4a204",
  processing: "#007cdb",
  finished: "#5dd402"
} as const satisfies Record<OperationStatus, string>

const OperationNode: BaseFC<{
  timestamp: number,
  operation: OperationState,
  pos: { x: number, y: number },
  radius: number,
  scaleRate: number
}> = (props) => {
  const scaleRateRef = useRef(props.scaleRate)
  useEffect(() => { scaleRateRef.current = props.scaleRate }, [props.scaleRate])

  return (
    <div className={props.className}
      css={css`
        width: ${props.radius * 2}px;
        height: ${props.radius * 2}px;
        border-radius: ${props.radius}px;
        border: 2px solid black;
        transform: translate(-50%, -50%);
        position: absolute;
        left: ${props.pos.x}px;
        top: ${props.pos.y}px;
        background-color: ${operationColor[operationStatusMapper[props.operation.status]]};
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1;
      `}
    >
      {props.operation.id}
      {
        (
          () => {
            const style = css`
              position: absolute;
              bottom: -15px;
            `
            switch (operationStatusMapper[props.operation.status]) {
              case "blocked":
              case "unscheduled":
              case "waiting":
                {
                  if (props.operation.predecessors.length !== 0) {
                    const total = props.operation.predecessors.length
                    const arrived = props.operation.arrived_preds.length
                    return <Progress percent={arrived / total * 100} steps={total} showInfo={false} css={style} />
                  }
                  break
                }
              case "processing":
                {
                  const total = props.operation.process_time
                  const rest = props.operation.finish_timestamp - props.timestamp
                  return <Progress percent={(1 - rest / total) * 100} showInfo={false} css={style} />
                }
              case "finished":
                {
                  if (props.operation.successors.length !== 0) {
                    const total = props.operation.successors.length
                    const sent = props.operation.sent_succs.length
                    return <Progress percent={sent / total * 100} steps={total} showInfo={false} css={style} />
                  }
                  break
                }
            }
          }
        )()
      }
    </div>
  )
}

const OperationLine: BaseFC<{
  timestamp: number,
  pState: OperationState,
  pPos: { x: number, y: number },
  sState: OperationState,
  sPos: { x: number, y: number },
  AGVinfo: { state: AGVState, dist: number } | undefined,
  nodeRadius: number,
  scaleRate: number
}> = (props) => {
  const [travelPercent, setTravelPercent] = useState<number | null>(null)
  useEffect(() => {
    if (props.AGVinfo === undefined) {
      setTravelPercent(null)
    }
    else {
      const total = props.AGVinfo.dist / props.AGVinfo.state.speed
      const rest = props.AGVinfo.state.finish_timestamp - props.timestamp
      console.log(
        props.pState.id,
        props.sState.id,
        props.AGVinfo.state.id,
        props.AGVinfo.state.position,
        props.AGVinfo.state.target_machine,
        props.AGVinfo.state.speed,
        props.AGVinfo.dist,
        rest
      )
      setTravelPercent((1 - rest / total) * 100)
    }
  }, [props.AGVinfo, props.timestamp])

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

  const raw_len = Math.hypot(props.sPos.x - props.pPos.x, props.sPos.y - props.pPos.y)
  const len = raw_len - props.nodeRadius * 2
  const rot = Math.asin((props.sPos.y - props.pPos.y) / raw_len)

  const center_x = (props.pPos.x + props.sPos.x) / 2 - len / 2
  const center_y = (props.pPos.y + props.sPos.y) / 2


  return (
    <>
      <div className={props.className}
        ref={refs.setReference} {...getReferenceProps()} css={css`
        width: ${len}px;
        border: 1px ${props.pState.id === 0 || props.sState.id === 9999 ? "dashed" : "solid"} black;
        position: absolute;
        rotate: ${rot}rad;
        left: calc(50% + ${center_x}px);
        top: calc(50% + ${center_y}px);
        z-index: 0;
        :hover {
          border-color: red;
          z-index: 4;
          /* & > * {
            color: red;
          } */
        }
      `}
      >
        <CaretRightFilled css={css`
          position: absolute;
          right: 0;
          translate: 40% -50%;
        `} />
        {
          props.AGVinfo && (
            <div css={css`
              width: 17px;
              height: 17px;
              background-color: black;
              border: 2px solid yellow;
              color: white;
              position: absolute;
              left: ${travelPercent}%;
              translate: -50% -50%;
              display: flex;
              justify-content: center;
              align-items: center;
            `}>
              {props.AGVinfo.state.id}
            </div>
          )
        }
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
          {`${props.pState.id}->${props.sState.id}`}
        </div>
      )}
    </>
  )
}

const OperationViewer: BaseFC<{
  state: EnvState,
  paths: Paths
}> = (props) => {
  const [operationInfoList, setOperationInfoList] = useState<{ id: number, pos: { x: number, y: number } }[]>([])

  const colSpace = 150
  const rowSpace = 125

  const nodeRadius = 25

  const fix_sort = () => {
    let predecessors_info: {
      id: number,
      remains: number[],
      preds_pos: number[]
    }[] = props.state.operations.map((item) => ({ id: item.id, remains: item.predecessors, preds_pos: [] }))

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
    const info_list: { id: number, pos: { x: number, y: number } }[] = []
    for (const [col_idx, col] of layers.entries()) {
      for (const [idx, id] of col.entries()) {
        info_list.push({
          id,
          pos: {
            x: (col_idx - layers.length / 2 + 0.5) * colSpace,
            y: (idx - col.length / 2 + 0.5) * rowSpace
          }
        })
      }
    }
    setOperationInfoList(info_list)
  }
  useEffect(fix_sort, [props.state.operations])

  const [graphOffset, setGraphOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })
  const [scaleRatio, setScaleRatio] = useState<number>(1)
  const resetPos = () => {
    setGraphOffset({ x: 0, y: 0 })
    setScaleRatio(1)
  }
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
        position: absolute;
        overflow: visible;
        top: calc(50% + ${graphOffset.y}px);
        left: calc(50% + ${graphOffset.x}px);
        scale: ${scaleRatio};
      `}>
        {
          operationInfoList.map(({ id, pos }) => (
            props.state.operations.find((item) => item.id === id)!.predecessors.map((pred_id) => (
              operationInfoList.find((item) => item.id === pred_id)!
            )).map((p) => (
              <OperationLine
                key={`${p.id}-${id}`}
                timestamp={props.state.timestamp}
                pState={props.state.operations.find((state) => state.id === p.id)!} pPos={p.pos}
                sState={props.state.operations.find((state) => state.id === id)!} sPos={pos}
                AGVinfo={
                  (
                    () => {
                      const agv = props.state.AGVs.find((agv) => (
                        AGVStatusMapper[agv.status] === "transporting"
                        && agv.loaded_item?.from === p.id
                        && agv.loaded_item.to === id
                        && agv.finish_timestamp > props.state.timestamp
                      ))
                      if (agv === undefined) {
                        return undefined
                      }
                      return {
                        state: agv,
                        dist: props.paths[agv.position][agv.target_machine][1]
                      }
                    }
                  )()
                }
                nodeRadius={nodeRadius} scaleRate={scaleRatio} />
            ))
          ))
        }
        {
          operationInfoList.map(({ id, pos }) => (
            <OperationNode key={id} timestamp={props.state.timestamp}
              operation={props.state.operations.find((item) => item.id === id)!}
              pos={pos} radius={nodeRadius} scaleRate={scaleRatio}
            />
          ))
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


const MachineNode: BaseFC<{
  state: MachineState,
  operation: OperationState | undefined
}> = (props) => {

  return (
    <div className={props.className}
      css={css`
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: gray;
        border: 2px solid black;
        display: flex;
        justify-content: center;
        align-items: center;
      `}
    >
      {props.state.id}
      {
        MachineStatusMapper[props.state.status] === "working" && (
          <>
            <div css={css`
              position: absolute;
              top: 0;
              right: 0;
              translate: 30% -30%;
              width: 25px;
              height: 25px;
              background-color: ${operationColor["processing"]};
              color: black;
              font-size: small;
              border-radius: 50%;
              border: 2px solid black;
              overflow: hidden;
              display: flex;
              justify-content: center;
              align-items: center;
            `}>
              {props.operation!.id}
            </div>
          </>
        )
      }
    </div>
  )
}

const MachinePath: BaseFC<{
  timestamp: number,
  fState: MachineState,
  tState: MachineState,
  scaleRatio: number,
  possibleAGVs: AGVState[],
  paths: Paths
}> = (props) => {
  const pos1 = props.fState.pos
  const pos2 = props.tState.pos

  const dx = pos1.x - pos2.x
  const dy = pos1.y - pos2.y
  const len = Math.hypot(dx, dy) * props.scaleRatio
  const rot = Math.PI - Math.atan2(dy, dx)

  return (
    <div className={props.className} css={css`
      position: absolute;
      width: ${len}px;
      border: 1px solid black;
      transform-origin: 0% 50%;
      rotate: ${rot}rad;
      left: ${pos1.x * props.scaleRatio}px;
      bottom: ${pos1.y * props.scaleRatio}px;
    `}
    >
      {/* TODO 添加对出于当前路径上的AGV的筛选和绘制*/}
    </div>
  )
}

const MachineViewer: BaseFC<{
  state: EnvState,
  paths: Paths
}> = (props) => {
  const [scaleRatio, setScaleRatio] = useState<number>(1)
  const [offset, setOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })

  const trackDrag = (e: MouseEvent) => {
    e.preventDefault()
    let prev_x = e.clientX
    let prev_y = e.clientY

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const dx = e.clientX - prev_x
      const dy = e.clientY - prev_y
      setOffset(({ x, y }) => ({ x: x + dx, y: y + dy }))
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
    setOffset(({ x: px, y: py }) => ({ x: px * rate, y: py * rate }))
  }

  return (
    <div className={props.className} onMouseDown={trackDrag} onWheel={updateScale} css={css`
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
        top: calc(50% + ${offset.y}px);
        left: calc(50% + ${offset.x}px);
      `}>
        {
          props.state.machines.map((state) => (
            <MachineNode key={state.id} state={state}
              operation={props.state.operations.find((op) => state.working_operation === op.id)}
              css={css`
                position: absolute;
                bottom: ${state.pos.y * scaleRatio}px;
                left: ${state.pos.x * scaleRatio}px;
                translate: -50% 50%;
                z-index: 10;
              `}
            />
          ))
        }
        {
          props.state.direct_paths.map(([f_id, t_id]) => {
            return (
              <MachinePath
                key={`${f_id}-${t_id}`} timestamp={props.state.timestamp}
                fState={props.state.machines.find((m) => m.id === f_id)!}
                tState={props.state.machines.find((m) => m.id === t_id)!}
                scaleRatio={scaleRatio}
                possibleAGVs={
                  props.state.AGVs.filter((agv) => (
                    AGVStatusMapper[agv.status] !== "idle"
                    && (
                      props.paths[agv.position][agv.target_machine][0].findIndex((idx) => idx === f_id)
                      - props.paths[agv.position][agv.target_machine][0].findIndex((idx) => idx === t_id)
                      === -1
                    )
                  ))
                }
                paths={props.paths}
              />
            )
          })
        }
      </div>
    </div>
  )
}

const EnvViewer: BaseFC<{ state: EnvState, onReture: () => void }> = (props) => {
  const [modelPaths, setModelPaths] = useState<string[]>([])
  const [selectedModelPath, setSelectedModelPath] = useState<string | null>(null)
  const [sampleCount, setSampleCount] = useState<number>(4)
  const [simCount, setSimCount] = useState<number>(20)

  const [paths, setPaths] = useState<Paths>({})
  const [records, setRecords] = useState<{ time: number, info: Action | string, state: EnvState }[]>([])
  const [viewIdx, setViewIdx] = useState<number | null>(null)

  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)
  const [abortController, setAbortController] = useState<AbortController | null>(null)
  const [progress, setProgress] = useState<[number, number, number]>([0, 0, 0])

  const [timelineItems, setTimelineItems] = useState<TimelineItemProps[]>([])
  const timelineItemRefs = useRef<HTMLAnchorElement[]>([])

  useEffect(() => {
    (async () => setModelPaths((await modelList())))();
    (async () => setPaths(await getPaths(props.state)))()
  }, [])

  const loadNewModel = async () => {
    const path = await open()
    if (path !== null) {
      if (modelPaths.find((p) => p === path) === undefined) {
        await loadModel(path)
        setModelPaths((prevs) => [...prevs, path])
        setSelectedModelPath(path)
      }
    }
  }

  const removeLoadedModel = async (path: string) => {
    await removeModel(path)
    setModelPaths((prevs) => prevs.filter((p) => p !== path))
    if (selectedModelPath === path) {
      setSelectedModelPath(null)
    }
  }

  const initProgress = async () => {
    setRecords([{ time: 0, info: "init", state: await initEnv(props.state) }])
    setViewIdx(null)
    setProgress([0, 0, 0])
  }


  const startPredict = async () => {
    await initProgress()
    const ac = new AbortController()
    setAbortController(ac)
    setIsModalOpen(true)
    try {
      await predict(
        props.state,
        selectedModelPath!,
        sampleCount,
        simCount,
        (progress) => {
          setRecords((prevs) => {
            if (!ac.signal.aborted) {
              return [
                ...prevs,
                {
                  time: progress.graph_state.timestamp,
                  info: progress.action,
                  state: progress.graph_state
                }
              ]
            }
            else {
              return [...prevs]
            }
          })
          setProgress([progress.round_count, progress.total_step, progress.finished_step])
        },
        ac.signal
      )
      await new Promise((resolve) => setTimeout(resolve, 400))
      setRecords((prevs) => {
        const lastState = prevs[prevs.length - 1].state
        return [
          ...prevs,
          {
            time: lastState.timestamp,
            info: "finished",
            state: lastState
          }
        ]
      })
    } catch (error) {
      setRecords((prevs) => {
        const lastState = prevs[prevs.length - 1].state
        return [
          ...prevs,
          {
            time: lastState.timestamp,
            info: "interrupted",
            state: lastState
          }
        ]
      })
    }

    setAbortController(null)
    setIsModalOpen(false)
  }

  useEffect(() => {
    if (records.length === 0) {
      setViewIdx(null)
    }
    else {
      setViewIdx(records.length - 1)
    }
  }, [records])

  useEffect(() => {
    let items: TimelineItemProps[] = []
    timelineItemRefs.current = []
    records.forEach((item, idx) => {
      if (typeof item.info === "string") {
        items.push({
          label: item.time.toFixed(2),
          color: idx === viewIdx ? "red" : undefined,
          children: (
            <a ref={(el) => timelineItemRefs.current[idx] = el!} css={css`
              font-weight: ${idx === viewIdx ? "bold" : "normal"};
            `} onClick={(e) => { e.preventDefault(); setViewIdx(idx) }}
            >
              {item.info}
            </a>
          )
        })
      }
      else {
        if (actionStatusMapper[item.info.action_type] === "wait") {
          items.push({
            label: item.time.toFixed(2),
            dot: <ClockCircleOutlined />,
            color: idx === viewIdx ? "red" : undefined,
            children: (
              <a ref={(el) => timelineItemRefs.current[idx] = el!} css={css`
                font-weight: ${idx === viewIdx ? "bold" : "normal"};
              `} onClick={(e) => { e.preventDefault(); setViewIdx(idx) }}
              >
                wait
              </a>
            )
          })
        }
        else {
          items.push({
            color: idx === viewIdx ? "red" : "gray",
            children: (
              <a ref={(el) => timelineItemRefs.current[idx] = el!} css={css`
                font-weight: ${idx === viewIdx ? "bold" : "normal"};
              `} onClick={(e) => { e.preventDefault(); setViewIdx(idx) }}
              >
                {
                  (() => {
                    const AGV = item.state.AGVs[item.info.AGV_id!]
                    switch (actionStatusMapper[(item.info as Action).action_type]) {
                      case "move":
                        return (
                          <div>
                            move
                            <Flex vertical>
                              <Typography.Text>id: {AGV.id}</Typography.Text>
                              <Typography.Text>path: {`${AGV.position} -> ${AGV.target_machine}`}</Typography.Text>
                            </Flex>
                          </div>
                        )
                      case "pick":
                        {
                          const item = AGV.target_item!
                          return (
                            <div>
                              pick
                              <Flex vertical>
                                <Typography.Text>id: {AGV.id}</Typography.Text>
                                <Typography.Text>path: {`${AGV.position} -> ${AGV.target_machine}`}</Typography.Text>
                                <Typography.Text>item: {`${item.from} -> ${item.to}`}</Typography.Text>
                              </Flex>
                            </div>
                          )
                        }
                      case "transport":
                        {
                          const item = AGV.loaded_item!
                          return (
                            <div>
                              transoport
                              <Flex vertical>
                                <Typography.Text>id: {AGV.id}</Typography.Text>
                                <Typography.Text>path: {`${AGV.position} -> ${AGV.target_machine}`}</Typography.Text>
                                <Typography.Text>item: {`${item.from} -> ${item.to}`}</Typography.Text>
                              </Flex>
                            </div>
                          )
                        }
                      default:
                        throw "wrong action_type"
                    }
                  })()
                }
              </a>
            )
          })
        }
      }
    })
    setTimelineItems(items)
  }, [viewIdx])

  useLayoutEffect(() => {
    if (viewIdx !== null) {
      const lastRecord = timelineItemRefs.current[viewIdx]
      lastRecord!.scrollIntoView({ behavior: "smooth", block: "center" })
    }
  }, [timelineItems])

  return (
    <>
      <Layout className={props.className}>
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
              模型：
              <Dropdown trigger={["click"]} menu={{
                items: [
                  ...modelPaths.map((path) => ({
                    label: (
                      <p onClick={() => setSelectedModelPath(path)}>
                        <Typography.Text>
                          {path}
                        </Typography.Text>
                        <Button
                          onClick={(e) => { e.preventDefault(); removeLoadedModel(path) }}
                          icon={<CloseOutlined />}
                        />
                      </p>
                    ),
                    key: path
                  })),
                  { type: "divider" },
                  { key: "new", label: <p onClick={loadNewModel}>加载模型</p> }
                ]
              }}>
                <a onClick={(e) => e.preventDefault()}>
                  {selectedModelPath?.split("\\").reverse()[0] || "未选择"}
                </a>
              </Dropdown>
              采样数量：
              <InputNumber min={1} value={sampleCount} onChange={(v) => setSampleCount(v || 1)} />
              模拟步数：
              <InputNumber min={0} value={simCount} onChange={(v) => setSimCount(v || 0)} />
              <Button type="primary" onClick={startPredict} disabled={selectedModelPath === null}>规划全部</Button>
              <Button css={css`
              margin-left: 10px;
            `} type="default" onClick={props.onReture}>返回</Button>
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
              <Splitter.Panel defaultSize="80%">
                <Splitter layout="vertical">
                  <Splitter.Panel defaultSize="70%">
                    <div css={css`
                      position: relative;
                      width: 100%;
                      height: 100%;
                    `}>
                      <OperationViewer
                        state={viewIdx !== null ? records[viewIdx].state : props.state}
                        paths={paths}
                        css={css`
                          width: 100%;
                          height: 100%;
                        `}
                      />
                      <Flex justify="center" gap={10} css={css`
                        position: absolute;
                        bottom: 10px;
                        width: 100%;
                      `}>
                        <Button shape="circle"
                          onClick={() => setViewIdx((prev) => prev! - 1)}
                          disabled={records.length === 0 || viewIdx === 0 || viewIdx === null}
                          icon={<LeftOutlined />}
                        />
                        <Button shape="circle"
                          onClick={() => setViewIdx((prev) => prev! + 1)}
                          disabled={records.length === 0 || viewIdx === records.length - 1 || viewIdx === null}
                          icon={<RightOutlined />}
                        />
                      </Flex>
                    </div>
                  </Splitter.Panel>
                  <Splitter.Panel min="20%">
                    <MachineViewer
                      state={viewIdx !== null ? records[viewIdx].state : props.state}
                      paths={paths}
                      css={css`
                        width: 100%;
                        height: 100%;
                      `}
                    />
                  </Splitter.Panel>
                </Splitter>
              </Splitter.Panel>
              <Splitter.Panel collapsible>
                {
                  viewIdx !== null ? (
                    <Timeline mode="left" items={timelineItems} css={css`
                      margin: 15px;
                    `} />
                  ) : (
                    <Flex justify="center" align="center" css={css`
                      height: 100%;
                    `}>
                      <Empty />
                    </Flex>
                  )
                }
              </Splitter.Panel>
            </Splitter>
          </Card>
        </Layout.Content>
      </Layout>
      <Modal open={isModalOpen} onCancel={() => abortController!.abort()} closable={false}
        footer={(_, { CancelBtn }) => <CancelBtn />}
      >
        {progress}
      </Modal>
    </>
  )
}

export default EnvViewer