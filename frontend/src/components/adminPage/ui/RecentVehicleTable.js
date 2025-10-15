import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function RecentVehicleTable({activityList}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Lượt xe ra/vào gần nhất</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead className="border-b">
            <tr className="text-left">
              <th className="py-2">Biển số</th>
              <th>Thời gian</th>
              <th>Trạng thái</th>
            </tr>
          </thead>
          <tbody>
            {activityList.map((a, i) => (
              <tr key={i} className="border-b last:border-none">
                <td className="py-2">{a.plate}</td>
                <td>{a.time}</td>
                <td
                  className={
                    a.status === "Vào" ? "font-medium text-green-600" : "font-medium text-red-500"
                  }
                >
                  {a.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
