package com.example.backend.controller;

import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/mockCapture")
@CrossOrigin(origins = "*")
public class MockCaptureController {

    @PostMapping("")
    public Map<String, String> captureOpenBarrier(@RequestBody Map<String, String> body){
        System.out.println("Capture nhận request: " + body);
        return Map.of("status", "received");
    }
}
