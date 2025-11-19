"use client";

import { ParkingSpotStatus } from "@/enums/ParkingSpotStatus.enum";
import React from "react";

export default function ParkingSpot({name, status, onclickhanlde }) {

  const bgColor =
    status === ParkingSpotStatus.AVAILABLE
      ? "bg-green-400"
      : "bg-red-400";

  return (
    <div
      className={`w-[100px] h-[100px] border-2 ${bgColor} rounded-lg shadow-md flex items-center justify-center text-lg font-semibold hover:scale-105 transition-transform cursor-pointer`}
      onClick={onclickhanlde}
    >
      {name}
    </div>
  );
}
