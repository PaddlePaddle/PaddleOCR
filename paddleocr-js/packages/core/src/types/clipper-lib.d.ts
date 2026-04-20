/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable @typescript-eslint/no-unused-vars */
declare module "clipper-lib" {
  interface IntPoint {
    X: number;
    Y: number;
  }

  type Path = IntPoint[];
  type Paths = Path[];

  class ClipperOffset {
    AddPath(path: Path, joinType: number, endType: number): void;
    Execute(result: Paths, delta: number): void;
  }

  const JoinType: {
    readonly jtRound: number;
  };

  const EndType: {
    readonly etClosedPolygon: number;
  };

  const ClipperLib: {
    ClipperOffset: typeof ClipperOffset;
    Paths: { new (): Paths };
    JoinType: typeof JoinType;
    EndType: typeof EndType;
  };

  export default ClipperLib;
}
