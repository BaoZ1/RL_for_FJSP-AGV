/** @jsxImportSource @emotion/react */

import { FC, useEffect, useState, useMemo, MouseEvent, WheelEvent, useRef } from "react"
import {
  Splitter, Button, Layout, Card, Empty, Flex, Modal,
  Form, InputNumber, Space, FormInstance, FloatButton, Select,
} from "antd"
import { open } from "@tauri-apps/plugin-dialog"
import { css } from "@emotion/react";
import {
  useFloating, useClientPoint, useInteractions,
  useHover, offset, safePolygon, arrow
} from '@floating-ui/react';
import {
  RedoOutlined, CaretRightFilled, AimOutlined,
  PlusCircleOutlined, CloseCircleOutlined, PlusOutlined
} from '@ant-design/icons';
import { OperationState, EnvState, GenerationParams, AGVState, MachineState, AddOperationParams } from "./types";
import { addOperation, addPath, loadEnv, newEnv, randEnv, removeOperation } from "./backend-api";

const OperationNode: FC<{
  className?: string,
  operation: { id: number, x: number, y: number },
  selected: boolean,
  onClick: () => void,
  onAddPredClick: () => void,
  onAddSuccClick: () => void,
  onRemoveClick: () => void,
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
        onClick={(e) => {
          e.stopPropagation()
          props.onClick()
        }}
        css={css`
          width: ${props.radius * 2}px;
          height: ${props.radius * 2}px;
          border-radius: ${props.radius}px;
          border: 2px ${type === "normal" ? "solid" : "dashed"} ${props.selected ? "red" : "black"};
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
      {
        isAddSuccOpen && type !== "end" && (
          <PlusCircleOutlined onClick={props.onAddSuccClick} css={css`
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
          <PlusCircleOutlined onClick={props.onAddPredClick} css={css`
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
          <CloseCircleOutlined onClick={props.onRemoveClick} css={css`
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

const OperationEditor: FC<{
  className?: string,
  states: OperationState[],
  selectedOperation: number | null,
  onBackgroundClick: () => void,
  onOperationClick: (id: number) => void,
  onAddOperation: (pred: number | null, succ: number | null) => void,
  onRemoveOperation: (id: number) => void
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
  useEffect(fix_sort, [props.states])

  const [graphOffset, setGraphOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })
  const [scaleRatio, setScaleRatio] = useState<number>(1)
  const resetPos = () => {
    setGraphOffset({ x: 0, y: 0 })
    setScaleRatio(1)
  }
  const [isDragging, setIsDragging] = useState(false)
  const trackGraphDrag = (e: MouseEvent) => {
    e.preventDefault()
    let prev_x = e.clientX
    let prev_y = e.clientY

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      setIsDragging(true)
      const dx = e.clientX - prev_x
      const dy = e.clientY - prev_y
      setGraphOffset(({ x, y }) => ({ x: x + dx, y: y + dy }))
      prev_x = e.clientX
      prev_y = e.clientY
    }
    const handleMouseUp = () => {
      setTimeout(() => setIsDragging(false), 100)
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
      <OperationNode key={operation.id} operation={operation} selected={operation.id === props.selectedOperation}
        radius={nodeRadius} scaleRate={scaleRatio}
        onClick={() => props.onOperationClick(operation.id)}
        onAddPredClick={() => props.onAddOperation(null, operation.id)}
        onAddSuccClick={() => props.onAddOperation(operation.id, null)}
        onRemoveClick={() => props.onRemoveOperation(operation.id)}
      />
    ))
  ], [operationPosList, scaleRatio, props.selectedOperation])

  return (
    <div className={props.className}
      onMouseDown={trackGraphDrag} onWheel={updateScale}
      onClick={() => {
        if (!isDragging) {
          props.onBackgroundClick()
        }
      }}
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

const MachineNode: FC<{
  className?: string,
  state: MachineState,
  selected: boolean,
  onClick: () => void,
  onDrag: (dx: number, dy: number) => void
}> = (props) => {

  const trackDrag = (e: MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()

    let prev_x = e.clientX
    let prev_y = e.clientY

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const dx = e.clientX - prev_x
      const dy = e.clientY - prev_y
      props.onDrag(dx, dy)
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

  return (
    <div className={props.className} onMouseDown={trackDrag}
      onClick={(e) => {
        e.stopPropagation()
        props.onClick()
      }}
      css={css`
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: gray;
        border: 2px solid ${props.selected ? "red" : "black"};
        display: flex;
        justify-content: center;
        align-items: center;

        &:hover {
          border-color: red;
        }
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

const MachineEditor: FC<{
  className?: string,
  states: MachineState[],
  paths: [number, number][],
  selected: number | null,
  onBackgroundClicked: () => void,
  onMachineClicked: (id: number) => void,
  onMachineDragged: (id: number, dx: number, dy: number) => void
}> = (props) => {
  const [scaleRatio, setScaleRatio] = useState<number>(1)
  const [offset, setOffset] = useState<{ x: number, y: number }>({ x: 0, y: 0 })

  const [isDragging, setIsDragging] = useState(false)

  const trackDrag = (e: MouseEvent) => {
    e.preventDefault()
    let prev_x = e.clientX
    let prev_y = e.clientY

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      setIsDragging(true)
      const dx = e.clientX - prev_x
      const dy = e.clientY - prev_y
      setOffset(({ x, y }) => ({ x: x + dx, y: y + dy }))
      prev_x = e.clientX
      prev_y = e.clientY
    }
    const handleMouseUp = () => {
      setTimeout(() => setIsDragging(false), 100)
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
    <div className={props.className} onMouseDown={trackDrag} onWheel={updateScale}
      onClick={()=>{
        if(!isDragging) {
          props.onBackgroundClicked()
        }
      }} css={css`
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
            <MachineNode key={state.id} state={state} selected={state.id === props.selected}
              onClick={() => props.onMachineClicked(state.id)}
              onDrag={(dx, dy) => props.onMachineDragged(state.id, dx / scaleRatio, dy / scaleRatio)}
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

const AGVNode: FC<{ state: AGVState }> = (props) => {
  const [reference, setReference] = useState<HTMLElement | null>(null)
  const arrowRef = useRef(null)

  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const { refs: infoRefs, floatingStyles: infoFloatingStyles, context: infoContexts, middlewareData } = useFloating({
    placement: "top",
    open: isInfoOpen,
    onOpenChange: setIsInfoOpen,
    elements: { reference },
    middleware: [offset(10), arrow({ element: arrowRef })]
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
            z-index: 10;
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
            `} />
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

const AddOperationConfigForm: FC<{
  formData: FormInstance<AddOperationParams>,
  onFinish: () => void,
  machines?: MachineState[]
}> = (props) => {
  const typeMap = new Map<number, number[]>()
  props.machines?.forEach((machine) => {
    typeMap.set(machine.type, [...(typeMap.get(machine.type) || []), machine.id])
  })

  const machineTypes = Array.from(typeMap.entries()).map(([type, machines]) => {
    return {
      label: `${type} (${machines})`,
      disabled: type === 0,
      value: type
    }
  })

  return (
    <Form form={props.formData} onFinish={props.onFinish}>
      <Flex justify="center" align="center" css={css`
          margin-bottom: 20px;
        `}
      >
        <div css={css`
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px dashed black;
            background-color: gray;
            opacity: ${props.formData.getFieldValue("succ") !== null ? "1" : "0.3"};
          `}
        />
        <div css={css`
            width: 40px;
            border: 1px solid black;
            opacity: ${props.formData.getFieldValue("succ") !== null ? "1" : "0.3"};
          `}
        />
        <div css={css`
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid black;
            background-color: gray;
            display: flex;
            justify-content: center;
            align-items: center;
          `}
        >
          {props.formData.getFieldValue("pred") || props.formData.getFieldValue("succ")}
        </div>
        <div css={css`
            width: 40px;
            border: 1px solid black;
            opacity: ${props.formData.getFieldValue("pred") !== null ? "1" : "0.3"};
          `}
        />
        <div css={css`
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px dashed black;
            background-color: gray;
            opacity: ${props.formData.getFieldValue("pred") !== null ? "1" : "0.3"};
          `}
        />
      </Flex>
      <Form.Item<AddOperationParams> name="machine_type" label="设备类型">
        <Select options={machineTypes} />
      </Form.Item>
      <Form.Item<AddOperationParams> name="process_time" label="处理时间">
        <InputNumber min={0} step={0.01} />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">确定</Button>
      </Form.Item>
    </Form>
  )
}

const OperationInfoForm: FC<{
  formData: FormInstance<OperationState>,
  onFinish: () => void
}> = (props) => {
  return (
    <Form form={props.formData} onFinish={props.onFinish}>
      <div>{props.formData.getFieldValue("id")}</div>
    </Form>
  )
}

const RandParamConfigForm: FC<{
  formData: FormInstance<GenerationParams>,
  onFinish: (values: GenerationParams) => void
}> = (props) => {

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
      <Form.Item<GenerationParams> name="machine_type_count" label="类型数量">
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

const EnvEditor: FC<{
  className?: string,
  state: EnvState | null,
  setState: React.Dispatch<React.SetStateAction<EnvState | null>>,
  onStart: ()=>void,
}> = ({ className, state, setState, onStart }) => {

  const [isRandModalOpen, setIsRandModalOpen] = useState(false)
  const [randParamsForm] = Form.useForm<GenerationParams>()

  const [isAddOperationModalOpen, setIsAddOperationModalOpen] = useState(false)
  const [addOperationParamsForm] = Form.useForm<AddOperationParams>()
  const handelAddOperation = (pred: number | null, succ: number | null) => {
    addOperationParamsForm.setFieldsValue({ pred, succ })
    setIsAddOperationModalOpen(true)
  }
  const handelRemoveOperation = async (id: number) => {
    setState(await removeOperation(state!, id))
  }

  const [selectedOperation, setSelectedOperation] = useState<number | null>(null)
  const [isOperationInfoModalOpen, setIsOperationInfoModalOpen] = useState(false)
  const [operationInfoForm] = Form.useForm<OperationState>()
  const handelOperationClick = async (id: number) => {
    if (selectedOperation === id) {
      setIsOperationInfoModalOpen(true)
    }
    else {
      setSelectedOperation(id)
    }
  }
  useEffect(() => {
    if (!isOperationInfoModalOpen) {
      setSelectedOperation(null)
    }
  }, [isOperationInfoModalOpen])

  const [selectedMachine, setSelectedMachine] = useState<number | null>(null)
  const handelMachineClick = async (id: number) => {
    if (selectedMachine === null) {
      setSelectedMachine(id)
    }
    else if (selectedMachine === id) {

    }
    else {
      setState(await addPath(state!, selectedMachine, id))
      setSelectedMachine(null)
    }
  }
  const updateMachinePos = (id: number, dx: number, dy: number) => {
    setSelectedMachine(null)
    setState((env) => {
      const ret = structuredClone(env!)
      const target = ret.machines.find((item) => item.id === id)!
      target.pos = { x: target.pos.x + dx, y: target.pos.y - dy }
      return ret
    })
  }

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
                  setState(await loadEnv(path))
                }
              }}>
                读取
              </Button>
              <Button type="primary" onClick={async () => setState(await newEnv())}>新建</Button>
              <Space.Compact>
                <Button type="primary" onClick={() => setIsRandModalOpen(true)}>随机</Button>
                <Button type="primary" icon={<RedoOutlined />}
                  onClick={async () => { setState(await randEnv(randParamsForm.getFieldsValue())) }}
                />
              </Space.Compact>
              <Button type="primary" disabled={state === null} onClick={onStart} css={css`
                margin-left: 10px;
                &:not([disabled]) {
                  > span {
                    position: relative;
                  }

                  &::before {
                    content: '';
                    background: linear-gradient(135deg, #6253e1, #04befe);
                    position: absolute;
                    inset: -1px;
                    opacity: 1;
                    transition: all 0.3s;
                    border-radius: inherit;
                  }

                  &:hover::before {
                    opacity: 0;
                  }
                }
              `}>开始</Button>
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
              state === null
                ?
                <Flex justify="center" align="center" css={css`
                    height: 100%;
                  `}
                >
                  <Empty>
                    <Button type="primary" onClick={async () => setState(await newEnv())}>新建</Button>
                  </Empty>
                </Flex>
                :
                <Splitter>
                  <Splitter.Panel defaultSize="70%">
                    <OperationEditor states={state.operations} selectedOperation={selectedOperation}
                      onBackgroundClick={() => setSelectedOperation(null)}
                      onOperationClick={handelOperationClick}
                      onAddOperation={handelAddOperation}
                      onRemoveOperation={handelRemoveOperation}
                      css={css`
                        width: 100%;
                        height: 100%;
                      `}
                    />
                  </Splitter.Panel>
                  <Splitter.Panel min="20%" collapsible>
                    <Splitter layout="vertical">
                      <Splitter.Panel defaultSize="70%">
                        <MachineEditor states={state.machines} paths={state.direct_paths}
                          selected={selectedMachine}
                          onBackgroundClicked={() => setSelectedMachine(null)}
                          onMachineClicked={handelMachineClick}
                          onMachineDragged={updateMachinePos} css={css`
                            width: 100%;
                            height: 100%;
                          `}
                        />
                      </Splitter.Panel>
                      <Splitter.Panel>
                        <Flex gap="middle" wrap css={css`
                          padding: 15px;
                        `}>
                          {
                            state.AGVs.map((v) => <AGVNode key={v.id} state={v} />)
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
                              <PlusOutlined />
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
      <Modal title="添加工序" open={isAddOperationModalOpen} footer={null}
        onCancel={() => setIsAddOperationModalOpen(false)}
      >
        <AddOperationConfigForm formData={addOperationParamsForm} machines={state?.machines}
          onFinish={async () => {
            setIsAddOperationModalOpen(false)
            setState(await addOperation(state!, addOperationParamsForm.getFieldsValue(true)))
          }}
        />
      </Modal>
      <Modal title="工序详情" open={isOperationInfoModalOpen} footer={null}
        onCancel={() => setIsOperationInfoModalOpen(false)}
      >
        <OperationInfoForm formData={operationInfoForm}
          onFinish={async () => {
            setIsOperationInfoModalOpen(false)
          }}
        />
      </Modal>
      <Modal title="参数设置" open={isRandModalOpen} footer={null} onCancel={() => setIsRandModalOpen(false)}>
        <RandParamConfigForm formData={randParamsForm}
          onFinish={async (values) => {
            setIsRandModalOpen(false)
            setState(await randEnv(values))
          }}
        />
      </Modal>
    </>
  )
}

export default EnvEditor
