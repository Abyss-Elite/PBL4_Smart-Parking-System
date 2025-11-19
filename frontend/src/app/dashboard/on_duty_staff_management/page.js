"use client";

import { useState, useEffect } from "react";
import WeekSelector from "@/components/on_duty_staff_management/WeekSelector";
import WeeklySchedule from "@/components/on_duty_staff_management/WeeklySchedule";
import { startOfWeek } from "date-fns";

export default function SchedulePage() {
  const [weekStart, setWeekStart] = useState(startOfWeek(new Date(), { weekStartsOn: 1 }));
  const [schedule, setSchedule] = useState([]);

  useEffect(() => {
    setSchedule([
      {
        date: "2025-11-11",
        shift: "08:00-17:30",
        employee: { id: 1, name: "Nguyen Van A", color: "bg-red-400" },
      },
      {
        date: "2025-11-11",
        shift: "17:30-24:00",
        employee: { id: 2, name: "Tran Thi B", color: "bg-blue-400" },
      },
    ]);
  }, [weekStart]);

  return (
    <div className="space-y-6 p-6">
      <WeekSelector initialStartDate="2025-07-07" onWeekChange={(start) => setWeekStart(start)} />
      <WeeklySchedule weekStart={weekStart} schedule={schedule} />
    </div>
  );
}
