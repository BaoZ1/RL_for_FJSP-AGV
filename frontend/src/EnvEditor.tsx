/** @jsxImportSource @emotion/react */

import { useEffect, useState, useMemo, MouseEvent, WheelEvent, useRef } from "react"
import {
  Splitter, Button, Layout, Card, Empty, Flex, Modal, Form, Popover,
  InputNumber, Space, FormInstance, FloatButton, Select, Checkbox, message,
} from "antd"
import { open, save } from "@tauri-apps/plugin-dialog"
import { css } from "@emotion/react";
import { useFloating, useClientPoint, useInteractions, useHover, offset, safePolygon } from '@floating-ui/react';
import {
  RedoOutlined, CaretRightFilled, AimOutlined,
  PlusCircleOutlined, CloseCircleOutlined, PlusOutlined
} from '@ant-design/icons';
import {
  BaseFC, OperationState, EnvState, GenerationParams,
  AGVState, MachineState, AddOperationParams, AddAGVParams,
  AddMachineParams
} from "./types";
import { 
  addAGV, addMachine, addOperation, addPath, loadEnv, 
  newEnv, randEnv, removeMachine, removeOperation, saveEnv 
} from "./backend-api";

const OperationNode: BaseFC<{
  state: OperationState,
  pos: { x: number, y: number },
  selected: boolean,
  onClick: () => void,
  onAddPredClick: () => void,
  onAddSuccClick: () => void,
  onRemoveClick: () => void,
  radius: number,
  scaleRatio: number
}> = (props) => {
  const type = ({ 0: "start", 9999: "end" } as const)[props.state.id] || "normal"

  const scaleRatioRef = useRef(props.scaleRatio)
  useEffect(() => { scaleRatioRef.current = props.scaleRatio }, [props.scaleRatio])

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
      const scaled_offset_offset = 5 / scaleRatioRef.current

      return {
        mainAxis: -raw_offset_x + raw_offset_x / scaleRatioRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_y + raw_offset_y / scaleRatioRef.current
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
      const scaled_offset_offset = 5 / scaleRatioRef.current

      return {
        mainAxis: -raw_offset_x + raw_offset_x / scaleRatioRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_y + raw_offset_y / scaleRatioRef.current
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
      const scaled_offset_offset = 5 / scaleRatioRef.current

      return {
        mainAxis: -raw_offset_y + raw_offset_y / scaleRatioRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_x + raw_offset_x / scaleRatioRef.current
      }
    })]
  });
  const removeHover = useHover(removeContext, { handleClose: safePolygon() });
  const { getFloatingProps: getRemoveFloatingProps } = useInteractions([removeHover]);

  const { getReferenceProps } = useInteractions([addSuccHover, addPredHover, removeHover])

  return (
    <>
      <Popover mouseEnterDelay={0} mouseLeaveDelay={0} title="详细信息" content={
        <>
          <p>machine type: {props.state.machine_type}</p>
          <p>process time: {props.state.process_time.toFixed(2)}</p>
        </>
      }>
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
            left: ${props.pos.x}px;
            top: ${props.pos.y}px;
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
          {props.state.id}
        </div>
      </Popover>
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

const OperationLine: BaseFC<{
  p: { id: number, x: number, y: number },
  s: { id: number, x: number, y: number },
  nodeRadius: number,
  scaleRatio: number
}> = (props) => {
  const [isOpen, setIsOpen] = useState(false);
  const scaleRatioRef = useRef(props.scaleRatio)
  useEffect(() => { scaleRatioRef.current = props.scaleRatio }, [props.scaleRatio])
  const { refs, floatingStyles, context } = useFloating({
    placement: "top",
    open: isOpen,
    onOpenChange: setIsOpen,
    middleware: [offset(({ x, y, rects: { floating: { width, height } } }) => {
      const top_offset_x = -width / 2
      const top_offset_y = height
      const raw_offset_x = x - top_offset_x
      const raw_offset_y = -y - top_offset_y
      const scaled_offset_offset = 10 / scaleRatioRef.current

      return {
        mainAxis: -raw_offset_y + raw_offset_y / scaleRatioRef.current + scaled_offset_offset,
        crossAxis: -raw_offset_x + raw_offset_x / scaleRatioRef.current
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

const OperationEditor: BaseFC<{
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
        <OperationLine key={`${p.id}-${s.id}`} p={p} s={s} nodeRadius={nodeRadius} scaleRatio={scaleRatio} />
      ))
    )),
    operationPosList.map((id_pos) => (
      <OperationNode key={id_pos.id}
        state={props.states.find((item) => item.id === id_pos.id)!}
        pos={{ x: id_pos.x, y: id_pos.y }}
        selected={id_pos.id === props.selectedOperation}
        radius={nodeRadius} scaleRatio={scaleRatio}
        onClick={() => props.onOperationClick(id_pos.id)}
        onAddPredClick={() => props.onAddOperation(null, id_pos.id)}
        onAddSuccClick={() => props.onAddOperation(id_pos.id, null)}
        onRemoveClick={() => props.onRemoveOperation(id_pos.id)}
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

const MachineNode: BaseFC<{
  state: MachineState,
  selected: boolean,
  scaleRatio: number,
  onClick: () => void,
  onDrag: (dx: number, dy: number) => void,
  onDragFinish: () => void,
  onAddMachine: (direction: { x: number, y: number }) => void,
}> = (props) => {
  const scaleRatioRef = useRef(props.scaleRatio)
  useEffect(() => { scaleRatioRef.current = props.scaleRatio }, [props.scaleRatio])

  const [addMachineBtnDirection, setAddMachineBtnDirection] = useState<{ x: number, y: number }>({ x: 1, y: 0 })

  const [reference, setReference] = useState<HTMLElement | null>(null)

  const [isAddMachineOpen, setIsAddMachineOpen] = useState(false);
  const {
    refs: addMachineRefs,
    floatingStyles: addMachineFloatingStyles,
    context: addMachineContext
  } = useFloating({
    placement: "bottom",
    open: isAddMachineOpen,
    onOpenChange: setIsAddMachineOpen,
    elements: { reference },
    middleware: [offset(({ rects }) => {
      const centerX = 0
      const centerY = -rects.reference.height / 2 - rects.floating.height / 2
      return {
        mainAxis: centerY + addMachineBtnDirection.y * 40,
        crossAxis: centerX + addMachineBtnDirection.x * 40
      };
    }, [addMachineBtnDirection])]
    // middleware: [offset(({ x, y, rects: { floating: { height } } }) => {
    //   const placement_offset_x = 0
    //   const placement_offset_y = -height / 2
    //   const raw_offset_x = x - placement_offset_x
    //   const raw_offset_y = y - placement_offset_y
    //   const scaled_offset_offset = 5 / scaleRatioRef.current

    //   return {
    //     mainAxis: -raw_offset_x + raw_offset_x / scaleRatioRef.current + scaled_offset_offset,
    //     crossAxis: -raw_offset_y + raw_offset_y / scaleRatioRef.current
    //   }
    // })]
  });
  const addMachineHover = useHover(addMachineContext, { handleClose: safePolygon() });
  const { getFloatingProps: getAddMachineFloatingProps } = useInteractions([addMachineHover]);

  const { getReferenceProps } = useInteractions([addMachineHover])

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
      setTimeout(() => {
        props.onDragFinish()
      }, 100);
      removeEventListener('mousemove', handleMouseMove)
      removeEventListener('mouseup', handleMouseUp)
    }
    addEventListener('mousemove', handleMouseMove)
    addEventListener('mouseup', handleMouseUp)
  }

  return (
    <>
      <Popover mouseEnterDelay={0} mouseLeaveDelay={0} title="详细信息" content={
        <>
          <p>type: {props.state.type}</p>
          <p>position: {`(${props.state.pos.x.toFixed(2)}, ${props.state.pos.y.toFixed(2)})`}</p>
        </>
      }>
        <div className={props.className} ref={setReference} {...getReferenceProps()}
          onMouseDown={trackDrag}
          onMouseMove={(e) => {
            const selfRect = reference!.getBoundingClientRect()
            const dx = e.clientX - (selfRect.x + selfRect.width / 2)
            const dy = e.clientY - (selfRect.y + selfRect.height / 2)
            const len = Math.hypot(dx, dy)
            setAddMachineBtnDirection({ x: dx / len, y: dy / len })
          }}
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
      </Popover>
      {
        isAddMachineOpen && (
          <PlusCircleOutlined ref={addMachineRefs.setFloating} {...getAddMachineFloatingProps()}
            style={addMachineFloatingStyles}
            onClick={() => props.onAddMachine(addMachineBtnDirection)}
            css={css`
              font-size: 25px;
              position: absolute;
              top: 50%;
              left: 50%;
              transform: translate(-50%, -50%);
              background-color: white;
              border-radius: 50%;
              color: #777777;
              background-color: white;
              z-index: 10;

              :hover {
                color: black;
              }
            `}
          />
        )
      }
    </>
  )
}

const MachinePath: BaseFC = (props) => {

  return (
    <div className={props.className} css={css`
        
      `}
    />
  )
}

const MachineEditor: BaseFC<{
  states: MachineState[],
  paths: [number, number][],
  selected: number | null,
  onBackgroundClicked: () => void,
  onMachineClicked: (id: number) => void,
  onMachineDragged: (id: number, dx: number, dy: number) => void,
  onAddMachine: (relativeId: number, direction: { x: number, y: number }) => void
}> = (props) => {
  const [scaleRatio, setScaleRatio] = useState<number>(5)
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
      onClick={() => {
        if (!isDragging) {
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
            <MachineNode key={state.id} state={state} selected={state.id === props.selected} scaleRatio={scaleRatio}
              onClick={() => props.onMachineClicked(state.id)}
              onDrag={(dx, dy) => props.onMachineDragged(state.id, dx / scaleRatio, dy / scaleRatio)}
              onDragFinish={props.onBackgroundClicked}
              onAddMachine={(direction) => props.onAddMachine(state.id, direction)}
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

const AGVNode: BaseFC<{ state: AGVState, onRemoveClick: () => void }> = (props) => {
  const [reference, setReference] = useState<HTMLElement | null>(null)

  const [isRemoveOpen, setIsRemoveOpen] = useState(false);
  const { refs: removeRefs, floatingStyles: removeFloatingStyles, context: removeContexts } = useFloating({
    placement: "bottom-end",
    open: isRemoveOpen,
    onOpenChange: setIsRemoveOpen,
    elements: { reference }
  });
  const removeHover = useHover(removeContexts, { handleClose: safePolygon() });
  const { getFloatingProps: getRemoveFloatingProps } = useInteractions([removeHover]);

  const { getReferenceProps } = useInteractions([removeHover])

  return (
    <>
      <Popover mouseEnterDelay={0} mouseLeaveDelay={0} title="详细信息" content={
        <>
          <p>speed: {props.state.speed.toFixed(2)}</p>
          <p>init_pos: {props.state.position}</p>
        </>
      }>
        <div className={props.className} ref={setReference} {...getReferenceProps()} css={css`
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
      </Popover>
      {
        isRemoveOpen && (
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

const AddOperationConfigForm: BaseFC<{
  formData: FormInstance<AddOperationParams>,
  onFinish: () => void,
  machines: MachineState[]
}> = (props) => {
  const typeMap = new Map<number, number[]>()
  props.machines.forEach((machine) => {
    typeMap.set(machine.type, [...(typeMap.get(machine.type) || []), machine.id])
  })

  const machineTypes = Array.from(typeMap.entries()).map(([type, machines]) => (
    {
      label: `${type} (${machines})`,
      disabled: type === 0,
      value: type
    }
  ))

  return (
    <Form className={props.className} form={props.formData} onFinish={props.onFinish}>
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

const AddMachineConfigForm: BaseFC<{
  formData: FormInstance<AddMachineParams>,
  onFinish: (values: AddMachineParams) => void,
  machines: MachineState[]
}> = (props) => {
  const new_type = props.machines.reduce((p, m) => Math.max(p, m.type) + 1, 0)

  const pos = [
    (props.formData.getFieldValue('x') as number).toFixed(2),
    (props.formData.getFieldValue('y') as number).toFixed(2)
  ]

  return (
    <Form className={props.className} form={props.formData} onFinish={props.onFinish}
      initialValues={{
        type: new_type
      }}
    >
      <Form.Item<AddMachineParams> label="坐标">
        {`( ${pos[0]}, ${pos[1]} )`}
      </Form.Item>
      <Form.Item<AddMachineParams> name="type" label="设备类型">
        <Select options={props.machines.map((m) => m.type)
          .filter((v, i, a) => a.indexOf(v) === i)
          .map((v) => ({
            label: `${v}`,
            value: v
          })).concat({ label: "新增", value: new_type })} />
      </Form.Item>
      <Form.Item<AddMachineParams> name="pathTo" label="连接设备">
        <Select mode="multiple" options={props.machines.map((m) => ({
          label: m.id,
          value: m.id
        }))} />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">确定</Button>
      </Form.Item>
    </Form>
  )
}

const AddAGVConfigForm: BaseFC<{
  formData: FormInstance<AddAGVParams>,
  onFinish: (values: AddAGVParams) => void,
  machines: MachineState[]
}> = (props) => {
  return (
    <Form className={props.className} form={props.formData} onFinish={props.onFinish}
      initialValues={{
        init_pos: 0,
        speed: 10
      }}
    >
      <Form.Item<AddAGVParams> name="init_pos" label="初始位置">
        <Select options={props.machines?.map((m) => (
          {
            label: `${m.id}`,
            value: m.id
          }
        ))} />
      </Form.Item>
      <Form.Item<AddAGVParams> name="speed" label="速度">
        <InputNumber min={0.01} max={50} step={0.01} />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">确定</Button>
      </Form.Item>
    </Form>
  )
}


const OperationInfoForm: BaseFC<{
  formData: FormInstance<OperationState>,
  onFinish: () => void
}> = (props) => {
  return (
    <Form className={props.className} form={props.formData} onFinish={props.onFinish}>
      <div>{props.formData.getFieldValue("id")}</div>
    </Form>
  )
}

const RandParamConfigForm: BaseFC<{
  formData: FormInstance<GenerationParams>,
  onFinish: (values: GenerationParams) => void
}> = (props) => {

  return (
    <Form className={props.className} form={props.formData} initialValues={{
      operation_count: 10,
      machine_count: 5,
      AGV_count: 5,
      machine_type_count: 3,
      min_transport_time: 8,
      max_transport_time: 15,
      min_max_speed_ratio: 0.8,
      min_process_time: 8,
      max_process_time: 15,
      simple_mode: false
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
      <Form.Item<GenerationParams> name="simple_mode" label="简单模式" valuePropName="checked">
        <Checkbox />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">确定</Button>
      </Form.Item>
    </Form>
  )
}

const EnvEditor: BaseFC<{
  state: EnvState | null,
  setState: React.Dispatch<React.SetStateAction<EnvState | null>>,
  onStart: () => void,
}> = (props) => {

  const [messageApi, contextHolder] = message.useMessage();

  const [isRandModalOpen, setIsRandModalOpen] = useState(false)
  const [randParamsForm] = Form.useForm<GenerationParams>()

  const [isAddOperationModalOpen, setIsAddOperationModalOpen] = useState(false)
  const [addOperationParamsForm] = Form.useForm<AddOperationParams>()
  const handelAddOperation = (pred: number | null, succ: number | null) => {
    addOperationParamsForm.setFieldsValue({ pred, succ })
    setIsAddOperationModalOpen(true)
  }
  const handelRemoveOperation = async (id: number) => {
    removeOperation(props.state!, id).then((res) => {
      props.setState(res)
    }).catch((e) => {
      messageApi.error(e)
    })
  }

  const [isAddmachineModalOpen, setIsAddMachineModalOpen] = useState(false)
  const [addMachineParamsForm] = Form.useForm<AddMachineParams>()
  const handelAddMachine = async (relativeId: number, direction: { x: number, y: number }) => {
    const relativePos = props.state!.machines.find((v) => v.id === relativeId)!.pos
    addMachineParamsForm.setFieldsValue({
      type: props.state!.machines.reduce((p, m) => Math.max(p, m.type) + 1, 0),
      x: relativePos.x + direction.x * 10,
      y: relativePos.y - direction.y * 10,
      pathTo: [relativeId]
    })
    setIsAddMachineModalOpen(true)
  }

  const [isAddAGVModalOpen, setIsAddAGVModalOpen] = useState(false)
  const [addAGVParamsForm] = Form.useForm<AddAGVParams>()
  const handelAddAGV = () => {
    setIsAddAGVModalOpen(true)
  }
  const handleRemoveAGV = (id: number) => {
    props.setState((env) => {
      const ret = structuredClone(env!)
      ret.AGVs = ret.AGVs.filter((agv) => agv.id !== id)
      return ret
    })
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
      removeMachine(props.state!, id).then((res) => {
        props.setState(res)
      }).catch((e) => {
        messageApi.error(e)
      })
      setSelectedMachine(null)
    }
    else {
      addPath(props.state!, selectedMachine, id).then((res) => {
        props.setState(res)
      }).catch((e) => {
        messageApi.error(e)
      })
      setSelectedMachine(null)
    }
  }
  const updateMachinePos = (id: number, dx: number, dy: number) => {
    setSelectedMachine(null)
    props.setState((env) => {
      const ret = structuredClone(env!)
      const target = ret.machines.find((item) => item.id === id)!
      target.pos = { x: target.pos.x + dx, y: target.pos.y - dy }
      return ret
    })
  }

  return (
    <>
      {contextHolder}
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
              <Button type="primary" disabled={props.state === null} onClick={async () => {
                const path = await save({
                  filters: [{ name: "Graph file", extensions: ["graph"] }]
                })
                if (path !== null) {
                  messageApi.open({ key: "save graph", type: "loading", content: "保存中..." })
                  saveEnv(path, props.state!).then(() => {
                    messageApi.open({ key: "save graph", type: "success", content: `已保存至${path}` })
                  }).catch((e) => {
                    messageApi.open({ key: "save graph", type: "error", content: `保存失败: ${e}` })
                  })

                }
              }}>
                保存
              </Button>
              <Button type="primary" onClick={async () => {
                const path = await open({
                  filters: [{ name: "Graph file", extensions: ["graph"] }]
                })
                if (path !== null) {
                  messageApi.open({ key: "load graph", type: "loading", content: "加载中..." })
                  loadEnv(path).then((res) => {
                    props.setState(res)
                    messageApi.open({ key: "load graph", type: "success", content: "加载完成" })
                  }).catch((e) => {
                    messageApi.open({ key: "load graph", type: "error", content: `加载失败: ${e}` })
                  })
                }
              }}>
                读取
              </Button>
              <Button type="primary" onClick={async () => {
                newEnv().then((res) => {
                  props.setState(res)
                }).catch((e) => {
                  messageApi.error(e)
                })
              }}>
                新建
              </Button>
              <Space.Compact>
                <Button type="primary" onClick={() => setIsRandModalOpen(true)}>随机</Button>
                <Button type="primary" icon={<RedoOutlined />}
                  onClick={async () => {
                    randEnv(randParamsForm.getFieldsValue()).then((res) => {
                      props.setState(res)
                    }).catch((e) => {
                      messageApi.error(e)
                    })
                  }}
                />
              </Space.Compact>
              <Button type="primary" disabled={props.state === null} onClick={props.onStart} css={css`
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
              `}>
                开始
              </Button>
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
              props.state === null ? (
                <Flex justify="center" align="center" css={css`
                    height: 100%;
                  `}
                >
                  <Empty>
                    <Button type="primary" onClick={async () => newEnv().then((res) => {
                      props.setState(res)
                    }).catch((e) => {
                      messageApi.error(e)
                    })}>新建</Button>
                  </Empty>
                </Flex>
              ) : (
                <Splitter>
                  <Splitter.Panel defaultSize="70%">
                    <OperationEditor states={props.state.operations} selectedOperation={selectedOperation}
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
                        <MachineEditor states={props.state.machines} paths={props.state.direct_paths}
                          selected={selectedMachine}
                          onBackgroundClicked={() => setSelectedMachine(null)}
                          onMachineClicked={handelMachineClick}
                          onMachineDragged={updateMachinePos}
                          onAddMachine={handelAddMachine}
                          css={css`
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
                            props.state.AGVs.map((v) => (
                              <AGVNode key={v.id} onRemoveClick={() => handleRemoveAGV(v.id)} state={v} />
                            ))
                          }
                          <div onClick={handelAddAGV} css={css`
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
              )
            }
          </Card>
        </Layout.Content>
      </Layout>
      {
        props.state && (
          <>
            <Modal title="添加工序" open={isAddOperationModalOpen} footer={null}
              onCancel={() => setIsAddOperationModalOpen(false)}
            >
              <AddOperationConfigForm formData={addOperationParamsForm} machines={props.state!.machines}
                onFinish={async () => {
                  setIsAddOperationModalOpen(false)
                  addOperation(props.state!, addOperationParamsForm.getFieldsValue(true)).then((res) => {
                    props.setState(res)
                  }).catch((e) => {
                    messageApi.error(e)
                  })
                }}
              />
            </Modal>
            <Modal title="添加设备" open={isAddmachineModalOpen} footer={null}
              onCancel={() => setIsAddMachineModalOpen(false)}
            >
              <AddMachineConfigForm formData={addMachineParamsForm} machines={props.state!.machines}
                onFinish={async () => {
                  setIsAddMachineModalOpen(false)
                  addMachine(props.state!, addMachineParamsForm.getFieldsValue(true)).then((res) => {
                    props.setState(res)
                  }).catch((e) => {
                    messageApi.error(e)
                  })
                }}
              />
            </Modal>
            <Modal title="添加运载车" open={isAddAGVModalOpen} footer={null}
              onCancel={() => setIsAddAGVModalOpen(false)}
            >
              <AddAGVConfigForm formData={addAGVParamsForm} machines={props.state!.machines}
                onFinish={(values) => {
                  setIsAddAGVModalOpen(false)
                  addAGV(props.state!, values).then((res) => {
                    props.setState(res)
                  }).catch((e) => {
                    messageApi.error(e)
                  })
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
          </>
        )
      }
      <Modal title="参数设置" open={isRandModalOpen} footer={null} onCancel={() => setIsRandModalOpen(false)}>
        <RandParamConfigForm formData={randParamsForm}
          onFinish={async (values) => {
            setIsRandModalOpen(false)
            randEnv(values).then((res) => {
              props.setState(res)
            }).catch((e) => {
              messageApi.error(e)
            })
          }}
        />
      </Modal>
    </>
  )
}

export default EnvEditor
