import { OverviewCards } from "@/components/parking/generalParking/OverviewCards";
import { ParkingSearchFilter } from "@/components/parking/generalParking/ParkingSearchFilter";
import { ParkingTable } from "@/components/parking/generalParking/ParkingTable";
import { useEffect, useState } from "react";
import { paidPublicParkingAPI } from "@/api/parking-lot/paidPublicParkingAPI";

export default function GeneralParking() {
  const [parkedCars, setParkedCars] = useState([]);
  const [searchLicensePlate, setSearchLicensePlate] = useState("");
  const [parkingCondition, setParkingCondition] = useState();
  const [numberParkingCondition, setNumberParkingCondition] = useState();

  useEffect(() => {
    const fetchData = async () => {
      const res = await paidPublicParkingAPI.getParkingCondition();
      const res1 = await paidPublicParkingAPI.getCarsInLot();
      setParkedCars(res1.data);
      setNumberParkingCondition(res.data);
      const data = res.data;
      const stats = [
        {
          title: "Xe đang đỗ",
          value: data.parkedCars,
          color: "bg-blue-100 text-blue-700",
        },
        {
          title: "Tổng số chỗ",
          value: data.totalSpots,
          color: "bg-gray-100 text-gray-700",
        },
        {
          title: "Còn trống",
          value: data.availableSpot,
          color: "bg-green-100 text-green-700",
        },
        {
          title: "Xe vào/ra hôm nay",
          value: `${data.carsInToday}/${data.carsOutToday}`,
          color: "bg-yellow-100 text-yellow-700",
        },
      ];
      setParkingCondition(stats);
    };
    fetchData();
  }, []);

  const fetchAllCars = async () => {
    const res = await paidPublicParkingAPI.getCarsInLot();
    setParkedCars(res.data);
  };

  const searchCars = async () => {
    if (!searchLicensePlate || searchLicensePlate.trim() === "") {
      fetchAllCars();
    } else {
      const res = await paidPublicParkingAPI.searchPlate(searchLicensePlate);
      setParkedCars(res.data);
    }
  };

  return (
    <div className="space-y-6 p-3">
      <OverviewCards parkingCondition={parkingCondition} />

      <div className="rounded-2xl bg-white p-3 shadow dark:bg-neutral-900">
        <ParkingSearchFilter
          searchLicensePlate={searchLicensePlate}
          setSearchLicensePlate={setSearchLicensePlate}
          onSearch={searchCars}
          onReset={fetchAllCars}
        />

        <ParkingTable parkedCars={parkedCars} />
      </div>
    </div>
  );
}
