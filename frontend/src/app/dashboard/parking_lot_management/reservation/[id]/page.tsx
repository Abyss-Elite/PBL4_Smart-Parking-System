"use client";

import carAPI from "@/api/car/carAPI";
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
  const data1 = {
    parking_spot_id: numericId,
    parking_spot_name: "A1",
    test: "test",
    day: [
      {
        date: "2025-11-10",
        bookings: [
          {
            booking_id: 1,
            start_time: "2025-11-10T09:00:00Z",
            end_time: "2025-11-10T11:30:00Z",
          },
          {
            booking_id: 2,
            start_time: "2025-11-10T12:00:00Z",
            end_time: "2025-11-10T15:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-11",
        bookings: [
          {
            booking_id: 5,
            start_time: "2025-11-11T09:00:00Z",
            end_time: "2025-11-11T11:30:00Z",
          },
          {
            booking_id: 6,
            start_time: "2025-11-11T12:00:00Z",
            end_time: "2025-11-11T15:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-12",
        bookings: [
          {
            booking_id: 7,
            start_time: "2025-11-12T09:00:00Z",
            end_time: "2025-11-12T11:30:00Z",
          },
          {
            booking_id: 3,
            start_time: "2025-11-12T12:00:00Z",
            end_time: "2025-11-12T15:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-13",
        bookings: [
          {
            booking_id: 4,
            start_time: "2025-11-13T08:00:00Z",
            end_time: "2025-11-13T11:30:00Z",
          },
          {
            booking_id: 8,
            start_time: "2025-11-13T15:00:00Z",
            end_time: "2025-11-13T20:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-14",
        bookings: [
          {
            booking_id: 10,
            start_time: "2025-11-14T09:00:00Z",
            end_time: "2025-11-14T11:30:00Z",
          },
          {
            booking_id: 9,
            start_time: "2025-11-14T12:00:00Z",
            end_time: "2025-11-14T15:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-15",
        bookings: [
          {
            booking_id: 12,
            start_time: "2025-11-15T09:00:00Z",
            end_time: "2025-11-15T11:30:00Z",
          },
          {
            booking_id: 11,
            start_time: "2025-11-15T12:00:00Z",
            end_time: "2025-11-15T15:00:00Z",
          },
        ],
      },
      {
        date: "2025-11-16",
        bookings: [
          {
            booking_id: 13,
            start_time: "2025-11-16T09:00:00Z",
            end_time: "2025-11-16T11:30:00Z",
          },
          {
            booking_id: 14,
            start_time: "2025-11-16T12:00:00Z",
            end_time: "2025-11-16T15:00:00Z",
          },
        ],
      },
    ],
  };

  const [spot, setSpot] = useState({
    parking_spot_id: numericId,
    parking_spot_name: "",
    day: [],
  })

  const [selectedBookingId, setSelectedBookingId] = useState<number|null>(null);
  const [weekStart, setWeekStart] = useState(startOfWeek(new Date(), { weekStartsOn: 1 }));
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const hours = Array.from({ length: 25 }, (_, i) => i);

  function timeToPosition(dateString: string) {
    const clean = dateString.replace("Z", "");
    const [datePart, timePart] = clean.split("T");
    const [hour, minute] = timePart.split(":").map(Number);
    const hourHeight = 64; 
    return hour * hourHeight + (minute / 60) * hourHeight;
  }

  useEffect(()=>{
    const fetchData = async () => {
      const date = new Date(weekStart);
      const day = String(date.getDate()).padStart(2, "0");
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const year = date.getFullYear();

      const formattedDay = `${year}-${month}-${day}`;
      const res = await carAPI.getReservatedSpot(numericId, formattedDay);
      // setSpot(res.data);
      setSpot(data1);
    }
    fetchData();
  },[id, weekStart]);
  console.log(weekStart);
  return (
    <>
      <div className="container mx-auto px-4 py-6">
        <WeekSelector initialStartDate="2025-07-07" onWeekChange={(start) => setWeekStart(start)} />
        <Card className="overflow-x-auto rounded-lg border border-gray-200 shadow-lg">
          <CardHeader className="mb-4 flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">Reserved Parking</CardTitle>
            <h2 className="text-blue-800">
              Selected parking lot: <b className="font-extrabold">{spot.parking_spot_name}</b>
            </h2>
          </CardHeader>
          
          <CardContent className="">
            <div className="w-full overflow-x-auto p-4 bg-white">
              <div className="grid grid-cols-8 border">
                <div className="border p-2 font-bold text-center">GMT+07</div>
                {days.map((d,index) => {
                  const date = new Date(weekStart);
                  date.setDate(date.getDate() + index);
                  const dayNumber = date.getDate();
                  return (
                  <div key={d} className="border p-2 font-bold text-center">
                    <p>{d}</p>
                    <h2 className="text-emerald-800">{dayNumber}</h2>
                  </div>
                )})}
              </div>

              <div className="grid grid-cols-8 relative">
                <div className="border relative">
                  {hours.map((h) => (
                    <div key={h} className="h-16 border-b relative">
                      <span className="absolute -top-2 left-2 text-xs bg-white px-1">{h}:00</span>
                    </div>
                  ))}
                </div>

                {days.map((day) => (
                  <div key={day} className="border relative">
                    {hours.map((_, i) => (
                      <div key={i} className="h-16 border-b"></div>
                    ))}

                    {spot.day
                      .filter((d) => {
                        const date = new Date(d.date);
                        const dayName = days[date.getDay()];
                        return dayName === day;
                      })
                      .flatMap((d) => d.bookings)
                      .map((booking) => {
                        const top = timeToPosition(booking.start_time);
                        const end = timeToPosition(booking.end_time);
                        const height = end - top;

                        return (
                          <div
                            key={booking.booking_id}
                            className="absolute left-1 right-1 bg-blue-600 text-white p-2 rounded-md shadow-md border border-blue-800"
                            onClick={()=> setSelectedBookingId(booking.booking_id)}
                            style={{
                              top,
                              height,
                            }}
                          >
                            <div className="font-semibold text-sm">Booking #{booking.booking_id}</div>
                            <div className="text-xs opacity-90">
                              {booking.start_time.slice(11, 16)} - {booking.end_time.slice(11, 16)}
                            </div>
                          </div>
                        );
                      })}
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      {selectedBookingId && (
        <BookingDetail
          bookingId={selectedBookingId}
          onClose={() => setSelectedBookingId(null)}
        />
      )}
    </>
  );
}

