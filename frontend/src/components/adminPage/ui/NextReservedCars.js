import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function NextReservedCars({ nextReservedCars }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Xe sắp tới giờ đặt</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead className="border-b">
            <tr className="text-left">
              <th className="py-2">Biển số</th>
              <th>Thời gian đã đặt</th>
            </tr>
          </thead>
          <tbody>
            {Array.isArray(nextReservedCars) &&
              nextReservedCars.map((car, index) => (
                <tr
                  key={index}
                  className={`border-b ${index === nextReservedCars.length - 1 ? "last:border-none" : ""}`}
                >
                  <td className="py-2">{car.plate}</td>
                  <td>{car.scheduledTimeIn}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
