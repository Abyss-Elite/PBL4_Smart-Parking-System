"use client";

import { useState, useEffect, use } from "react";
import WeekSelector from "@/components/common/WeekSelector";
import WeeklySchedule from "@/components/on_duty_staff_management/WeeklySchedule";
import { set, startOfWeek } from "date-fns";
import { staffManagementAPI } from "@/api/staff-management/staff-managementAPI";
import { parkingLotAPI } from "@/api/parking-lot/parkingLotAPI";
import { formatDateOnly } from "@/utils/formatDateOnly";

export default function SchedulePage() {
  const [weekStart, setWeekStart] = useState(startOfWeek(new Date(), { weekStartsOn: 1 }));
  const [schedule, setSchedule] = useState([]);
  const [initialStartDate, setInitialStartDate] = useState();

  useEffect(() => {
    const fetchStaffManagementData = async () => {
      try {
        const res = await staffManagementAPI.getStaffList();
        setSchedule(res.data);
      } catch (err) {
        console.error("Lỗi API lịch trực:", err);
      }
    };
    fetchStaffManagementData();
  }, [weekStart]);

  useEffect(() => {
    const fetchInitialStartDateDta = async () => {
      try {
        const res = await parkingLotAPI.getInitialStartDate();
        setInitialStartDate(formatDateOnly(res.data.firstActiveDate));
      } catch (err) {
        console.error("Lỗi API ngày bắt đầu ban đầu:", err);
      }
    };
    fetchInitialStartDateDta();
  }, [initialStartDate]);

  return (
    <div className="space-y-6 p-6">
      <WeekSelector
        initialStartDate={initialStartDate}
        onWeekChange={(start) => setWeekStart(start)}
      />
      <WeeklySchedule weekStart={weekStart} schedule={schedule} />
    </div>
  );
}
