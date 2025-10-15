package com.example.backend.service;

import com.example.backend.model.Car;
import com.example.backend.repository.CarRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.util.Timeout;
import java.net.URL;
import java.net.HttpURLConnection;
import java.util.List;
import java.util.Optional;

@Service
public class CarService {

    private final RestTemplate restTemplate;

    @Autowired
    private CarRepository carRepository;

     public CarService() {
        RequestConfig config = RequestConfig.custom()
        .setConnectTimeout(Timeout.ofSeconds(3))
        .setResponseTimeout(Timeout.ofSeconds(3))
        .build();

        CloseableHttpClient client = HttpClients.custom()
        .setDefaultRequestConfig(config)
        .disableAutomaticRetries() 
        .build();

        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(client);

        this.restTemplate = new RestTemplate(factory);
    }


    public List<Car> getAllCars() {
        return carRepository.findAll();
    }

    public Car getCar(Long id) {
        Optional<Car> carOp = carRepository.findById(id);
        return carOp.orElse(null);
    }

    public Car createCar(Car car) {
        return carRepository.save(car);
    }

    public Car updateCar(Long id, Car carDetails) {
        Car car = carRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Car not found"));

        if (carDetails.getLicensePlate() != null && !carDetails.getLicensePlate().isBlank()) {
            car.setLicensePlate(carDetails.getLicensePlate());
        }
        if (carDetails.getDescription() != null) {
            car.setDescription(carDetails.getDescription());
        }
        if (carDetails.getIsOut() != null) {
            car.setIsOut(carDetails.getIsOut());
        }
        if (carDetails.getImageUrl() != null) {
            car.setImageUrl(carDetails.getImageUrl());
        }
        if (carDetails.getStatus() != null && !carDetails.getStatus().isBlank()) {
            car.setStatus(carDetails.getStatus());
        }
        if (carDetails.getRegistrationDate() != null) {
            car.setRegistrationDate(carDetails.getRegistrationDate());
        }
        if (carDetails.getIsDelete() != null) {
            car.setIsDelete(carDetails.getIsDelete());
        }
        if (carDetails.getLastTime() != null) {
            car.setLastTime(carDetails.getLastTime());
        }

        return carRepository.save(car);
    }

    public void deleteCar(Long id) {
        Car car = carRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Xe không tồn tại, không thể xóa"));
        carRepository.delete(car);
    }

    public List<Car> getCarsByUserId(Long userId) {
        return carRepository.findByOwner_Id(userId);
    }

    public boolean sendOpenSignalToCapture(String captureUrl, String plate) {
        try {
            new Thread(() -> {
                try {
                    URL url = new URL(captureUrl);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setRequestProperty("Content-Type", "application/json");
                    conn.setConnectTimeout(2000); // chỉ cần timeout ngắn
                    conn.setDoOutput(true);

                    String body = "{\"status\":\"OK\",\"licensePlate\":\"" + plate + "\"}";
                    conn.getOutputStream().write(body.getBytes());
                    conn.getOutputStream().flush();
                    conn.disconnect();

                    System.out.println("✅ Đã gửi tín hiệu mở cho ESP: " + captureUrl);
                } catch (Exception e) {
                    System.err.println("⚠️ Lỗi khi gửi tín hiệu: " + e.getMessage());
                }
            }).start();

            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }




    public Car getCarsByLicensePlateNumber(String plate, String captureUrl) {
        System.out.println(plate);
        Car car = carRepository.findByLicensePlate(plate);
        if (car == null) return null;

        boolean sent = sendOpenSignalToCapture(captureUrl, plate);
        System.out.println("Send to capture: " + sent);

        return car;
    }
}
