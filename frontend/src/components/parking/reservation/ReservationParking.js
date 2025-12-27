"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ParkingSpot from "@/components/parking/ParkingSpot";
import { useRouter } from "next/navigation";
import PATH from "@/routes/PATH";
import React, { useEffect, useState } from "react";
import { carAPI } from "@/api/car/carAPI";

export default function ReservationParking() {
  const [allSpot, setAllSpot] = useState([]);
  const router = useRouter();
  const handleClick = (id) => {
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.RESERVATION.DETAIL(id));
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await carAPI.getAllSpotReservation();
        setAllSpot(res.data);
      } catch (error) {
        console.error("Error fetching spots:", error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="container mx-auto px-4 py-6">
      <Card className="overflow-x-auto rounded-lg border border-gray-200 shadow-lg">
        <CardHeader className="mb-4 flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">Reserved Parking</CardTitle>
        </CardHeader>

        <CardContent className="grid grid-cols-3 justify-items-center gap-6 md:grid-cols-5">
          {allSpot.map((spot) => (
            <ParkingSpot
              key={spot.id}
              id={spot.id}
              name={spot.name}
              status={spot.status}
              onclickhanlde={() => handleClick(spot.id)}
            />
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
