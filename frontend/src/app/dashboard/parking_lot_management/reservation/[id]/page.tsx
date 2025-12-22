"use client";

import { carAPI } from "@/api/car/carAPI";
import MonthYearSelector from "@/components/common/MonthYearSelector";
import WeekSelector from "@/components/common/WeekSelector";
import BookingDetail from "@/components/parking/BookingDetail";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { startOfWeek } from "date-fns";
import React, { useEffect, useState } from "react";

export default function ReservationDetailManagementPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = React.use(params);
  const numericId = Number(id);
  // const data1 = {
  //   parkingSpotId: numericId,
  //   parkingSpotName: "A1",
  //   days: [
  //     {
  //       date: "2025-11-01",
  //       bookings: [
  //         {
  //           booking_id: 1,
  //         },
  //       ],
  //     },
  //     {
  //       date: "2025-11-02",
  //       bookings: [
  //         {
  //           booking_id: 5,
  //         },
  //       ],
  //     },
  //     {
  //       date: "2025-11-25",
  //       bookings: [
  //         {
  //           booking_id: 12,
  //         },
  //       ],
  //     },
  //     {
  //       date: "2025-11-29",
  //       bookings: [
  //         {
  //           booking_id: 13,
  //         },
  //       ],
  //     },
  //   ],
  // };

  const [spot, setSpot] = useState({
    parkingSpotId: numericId,
    parkingSpotName: "",
    days: [],
  })

  const [selectedBookingId, setSelectedBookingId] = useState<number|null>(null);
  const [daysInMonth, setDaysInMonth] = useState<number>(0)
  const [month, setMonth] = useState<number>(12)
  const [year, setYear] =useState<number>(2025)
  const [startDay, setStartDay] = useState<number>(0)
  const [date, setDate] = useState<string>()
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const hasBookingDay = (day:number) => {
    const dateStr = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    return spot.days.some ((d :any )=> d.date === dateStr && d.bookings.length>0)
  }
  const getBookingByDay = (day: number) => {
    const dateStr = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    const dayData = spot.days.find((d: any) => d.date === dateStr);
    return {
      bookingId: dayData?.bookings?.[0]?.bookingId ?? null,
      date: dateStr
    };
  };
  console.log("selectedBookingId", selectedBookingId);


  useEffect(() => {
    const fetchData = async () => {
      const firstDayOfMonth = new Date(year, month - 1, 1);
      setStartDay(firstDayOfMonth.getDay());
      const res = await carAPI.getReservatedSpot(numericId, month, year);
      setSpot(res.data);
      // setSpot(data1);
    }
    fetchData();
  },[id, month, year]);

  return (
    <>
      <div className="container mx-auto px-4 py-6">
        <MonthYearSelector onChange={({month, year, daysInMonth})=> {
          setMonth(month); 
          setYear(year); 
          setDaysInMonth(daysInMonth)
        }}/>
        <Card className="overflow-x-auto rounded-lg border border-gray-200 shadow-lg">
          <CardHeader className="mb-4 flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">Reserved Parking</CardTitle>
            <h2 className="text-blue-800">
              Selected parking lot: <b className="font-extrabold">{spot.parkingSpotName}</b>
            </h2>
          </CardHeader>

          <CardContent className="">
            <div className="w-full bg-white p-4 rounded-xl shadow">
              <div className="grid grid-cols-7 text-center font-semibold text-gray-500 mb-2">
                {days.map((d) => (
                  <div key={d}>{d}</div>
                ))}
              </div>

              <div className="grid grid-cols-7 gap-y-2 text-center">
                {Array.from({ length: startDay }).map((_, i) => (
                  <div key={`empty-${i}`} />
                ))}

                {Array.from({ length: daysInMonth }, (_, i) => {
                  const day = i + 1;
                  return (
                    <div
                      key={day}
                      onClick={() => {
                        const { bookingId, date } = getBookingByDay(day);
                          if (!bookingId) return;

                          setDate(date);
                          setSelectedBookingId(bookingId);
                      }}
                      className={`h-15 flex items-center justify-center rounded-full
                                cursor-pointer ${hasBookingDay(day) ? "bg-blue-500 text-white hover:bg-blue-600": "hover:bg-gray-100"}`}
                              
                    >
                      {day}
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      {selectedBookingId!==null && (
        <BookingDetail
          bookingId={selectedBookingId}
          spotId = {id}
          date = {date}
          onClose={() => setSelectedBookingId(null)}
        />
      )}
    </>
  );
}
