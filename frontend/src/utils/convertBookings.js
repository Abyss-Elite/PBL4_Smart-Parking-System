import { formatDateOnly } from "./formatDateOnly";

export function convertBookings(input) {
  const result = [];

  input.forEach((booking) => {
    booking.cars?.forEach((car) => {
      result.push({
        id: car.parkingSpot?.id ?? null,
        name: car.parkingSpot?.name ?? null,
        startTimeBooking: formatDateOnly(car.startTimeBooking) ?? null,
        endTimeBooking: formatDateOnly(car.endTimeBooking) ?? null,
      });
    });
  });

  return result;
}
