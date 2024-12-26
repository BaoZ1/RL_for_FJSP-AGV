/** @jsxImportSource @emotion/react */

import { FC, useEffect, useState, MouseEvent, WheelEvent, useRef } from "react"
import { Button, Card, Flex, FloatButton, Layout, Splitter } from "antd"
import { AGVState, EnvState, MachineState, OperationState } from "./types"
import { css } from "@emotion/react"
import { offset, useClientPoint, useFloating, useHover, useInteractions } from "@floating-ui/react"
import { AimOutlined, CaretRightFilled } from '@ant-design/icons'


const OperationNode: FC<{
  className?: string,
  operation: { id: number, x: number, y: number },
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
        left: ${props.operation.x}px;
        top: ${props.operation.y}px;
        background-color: wheat;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1;
        :hover {
          border-color: red;
        }
      `}
    >
      {props.operation.id}
    </div>
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

const OperationViewer: FC<{
  className?: string,
  states: OperationState[],
  AGVs: AGVState[],
}> = (props) => {
  const [operationPosList, setOperationPosList] = useState<{ id: number, x: number, y: number }[]>([])

  const colSpace = 150
  const rowSpace = 125

  const nodeRadius = 25

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
  useEffect(fix_sort, [])

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
        /* border: 5px solid red; */
        position: absolute;
        overflow: visible;
        top: calc(50% + ${graphOffset.y}px);
        left: calc(50% + ${graphOffset.x}px);
        scale: ${scaleRatio};
      `}>
        {
          operationPosList.map((s) => (
            props.states.find((item) => item.id === s.id)!.predecessors.map((pred_id) => (
              operationPosList.find((item) => item.id === pred_id)!
            )).map((p) => (
              <OperationLine key={`${p.id}-${s.id}`} p={p} s={s} nodeRadius={nodeRadius} scaleRate={scaleRatio} />
            ))
          ))
        }
        {
          operationPosList.map((operation) => (
            <OperationNode key={operation.id} operation={operation}
              radius={nodeRadius} scaleRate={scaleRatio}
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


const MachineNode: FC<{
  className?: string,
  state: MachineState,
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
    </div>
  )
}

const MachinePath: FC<{
  className?: string
}> = (props) => {

  return (
    <div className={props.className} css={css`
        
      `}
    />
  )
}

const MachineViewer: FC<{
  className?: string,
  states: MachineState[],
  paths: [number, number][],
  AGVs: AGVState[],
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
          props.states.map((state) => (
            <MachineNode key={state.id} state={state}
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
          props.paths.map(([f_id, t_id]) => {
            const pos1 = props.states.find((item) => item.id == f_id)!.pos
            const pos2 = props.states.find((item) => item.id == t_id)!.pos

            const dx = pos1.x - pos2.x
            const dy = pos1.y - pos2.y
            const len = Math.hypot(dx, dy) * scaleRatio
            const rot = Math.PI - Math.atan2(dy, dx)
            return (
              <MachinePath key={`${f_id}-${t_id}`} css={css`
                    position: absolute;
                    width: ${len}px;
                    border: 1px solid black;
                    transform-origin: 0% 50%;
                    rotate: ${rot}rad;
                    left: ${pos1.x * scaleRatio}px;
                    bottom: ${pos1.y * scaleRatio}px;
                  `}
              />
            )
          })
        }
      </div>
    </div>
  )
}

const EnvViewer: FC<{ className?: string, state: EnvState, onReture: () => void }> = (props) => {

  return (
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
            <Button type="primary" onClick={props.onReture}>返回</Button>
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
              <OperationViewer states={props.state.operations} AGVs={props.state.AGVs}
                css={css`
                  width: 100%;
                  height: 100%;
                `}
              />
            </Splitter.Panel>
            <Splitter.Panel min="20%" collapsible>
              <MachineViewer states={props.state.machines} paths={props.state.direct_paths}
                  AGVs={props.state.AGVs} css={css`
                    width: 100%;
                    height: 100%;
                  `}
              />
            </Splitter.Panel>
          </Splitter>
        </Card>
      </Layout.Content>
    </Layout>
  )
}

export default EnvViewer