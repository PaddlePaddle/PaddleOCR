// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.paddle.ocr.benchmark

import android.os.Build
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.model.OCRRunResult
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.pow
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class OCRBenchmarkTest {

    companion object {
        private const val TAG = "OCRBenchmark"

        // Instrumentation arguments (passed via -Pandroid.testInstrumentationRunnerArguments.*)
        private const val ARG_WARMUP = "warmup"
        private const val ARG_ITERATIONS = "iterations"
        private const val ARG_REC_BATCH_SIZE = "rec_batch_size"

        private const val DEFAULT_WARMUP = 3
        private const val DEFAULT_ITERATIONS = 10

        private lateinit var ocr: PaddleOCR
        private lateinit var outputDir: File

        @JvmStatic
        @BeforeClass
        fun setUp() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            OpenCVUtils.init(context)

            outputDir = File(context.getExternalFilesDir(null), "ocr_benchmark")
            outputDir.mkdirs()
            println("[OCRBenchmark] Output dir: ${outputDir.absolutePath}")

            runBlocking {
                val batchSize = getInstrumentationArgument(ARG_REC_BATCH_SIZE)?.toIntOrNull() ?: 1
                val config = PaddleOCRConfig(recBatchSize = batchSize)
                ocr = PaddleOCR.create(context, config, EngineConfig())
            }
            println("[OCRBenchmark] Engine ready, coldLoadTimeMs=${ocr.coldLoadTimeMs}")
        }

        private fun getInstrumentationArgument(key: String): String? {
            val args = InstrumentationRegistry.getArguments()
            return args[key] as? String
        }

        private fun saveAndPrint(filename: String, json: JSONObject) {
            val file = File(outputDir, filename)
            file.writeText(json.toString(2))
            println("[OCRBenchmark] Saved: ${file.absolutePath}")
            println("[OCRBenchmark] ---- $filename ----")
            json.toString(2).lines().forEach { println("[OCRBenchmark] $it") }
            println("[OCRBenchmark] ---- end ----")
        }
    }

    // ========================================================================
    // Accuracy
    // ========================================================================

    @Test
    fun testOCRExportJSON() {
        val imageBytes = loadBenchmarkImageBytes()
        val result = runBlocking { ocr.recognize(imageBytes) }
        val payload = buildExportPayload(result)
        saveAndPrint("accuracy_export.json", payload)

        assertTrue("Should have results", payload.getJSONArray("items").length() > 0)
        assertTrue("Should detect text", result.lineCount > 0)
    }

    // ========================================================================
    // Latency
    // ========================================================================

    @Test
    fun testLatencyBenchmark() {
        val imageBytes = loadBenchmarkImageBytes()
        val warmup = getInstrumentationArgument(ARG_WARMUP)?.toIntOrNull() ?: DEFAULT_WARMUP
        val iterations = getInstrumentationArgument(ARG_ITERATIONS)?.toIntOrNull() ?: DEFAULT_ITERATIONS

        println("[OCRBenchmark] Warming up $warmup iterations...")
        runBlocking { repeat(warmup) { ocr.recognize(imageBytes) } }

        println("[OCRBenchmark] Measuring $iterations iterations...")
        val totals = mutableListOf<Long>()
        val dets = mutableListOf<Long>()
        val detPre = mutableListOf<Long>()
        val detInf = mutableListOf<Long>()
        val detPost = mutableListOf<Long>()
        val recs = mutableListOf<Long>()
        val recPre = mutableListOf<Long>()
        val recInf = mutableListOf<Long>()
        val recPost = mutableListOf<Long>()
        val overheads = mutableListOf<Long>()
        val perLineMs = mutableListOf<Long>()
        val detShapes = mutableListOf<List<Int>>()
        val recShapes = mutableListOf<List<Int>>()
        var firstLineCount = 0

        runBlocking {
            repeat(iterations) { i ->
                val r = ocr.recognize(imageBytes)
                if (i == 0) firstLineCount = r.lineCount
                totals.add(r.totalTimeMs)
                dets.add(r.detectionTimeMs)
                detPre.add(r.detPreprocessMs)
                detInf.add(r.detInferenceMs)
                detPost.add(r.detPostprocessMs)
                recs.add(r.recognitionTimeMs)
                recPre.add(r.recPreprocessMs)
                recInf.add(r.recInferenceMs)
                recPost.add(r.recPostprocessMs)
                overheads.add(r.pipelineOverheadMs)
                detShapes.add(r.detInputShape)
                recShapes.addAll(r.recInputShapes)
                perLineMs.addAll(r.perLineRecMs)
                if ((i + 1) % 5 == 0) println("[OCRBenchmark]   $i/${iterations}...")
            }
        }

        val memInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memInfo)

        val payload = JSONObject().apply {
            put("schemaVersion", 1)
            put("deviceModel", Build.MODEL)
            put("osVersion", Build.VERSION.RELEASE)
            put("ortExecutionProvider", "CPU")
            put("inputImage", BenchmarkFixtures.defaultReferenceImageName)
            put("inputBytes", imageBytes.size)
            put("warmupIterations", warmup)
            put("measuredIterations", iterations)
            put("firstMeasuredLineCount", firstLineCount)
            put("coldLoadTimeMs", ocr.coldLoadTimeMs)
            if (totals.isNotEmpty()) {
                put("firstMeasuredRun", JSONObject().apply {
                    put("totalTimeMs", totals.first())
                    put("detectionTimeMs", dets.first())
                    put("recognitionTimeMs", recs.first())
                    put("pipelineOverheadTimeMs", overheads.first())
                })
            }
            put("totalTimeMs", summarizeMs(totals).toJson())
            put("detectionTimeMs", summarizeMs(dets).toJson())
            put("detectionPreprocessTimeMs", summarizeMs(detPre).toJson())
            put("detectionInferenceTimeMs", summarizeMs(detInf).toJson())
            put("detectionPostprocessTimeMs", summarizeMs(detPost).toJson())
            put("recognitionTimeMs", summarizeMs(recs).toJson())
            put("recognitionPreprocessTimeMs", summarizeMs(recPre).toJson())
            put("recognitionInferenceTimeMs", summarizeMs(recInf).toJson())
            put("recognitionPostprocessTimeMs", summarizeMs(recPost).toJson())
            put("pipelineOverheadTimeMs", summarizeMs(overheads).toJson())
            put("memoryPssKb", memInfo.totalPss)
            put("inputShapeDistribution", JSONObject().apply {
                put("detection", summarizeShapes(detShapes))
                put("recognition", summarizeShapes(recShapes))
            })
            if (perLineMs.isNotEmpty()) {
                put("recognitionPerLine", JSONObject().apply {
                    put("pooled", JSONObject().apply {
                        put("count", perLineMs.size)
                        put("totalMs", summarizeMs(perLineMs).toJson())
                    })
                })
            }
        }

        saveAndPrint("latency_benchmark.json", payload)
        printSpeedSummary(warmup, iterations, totals, dets, detPre, detInf, detPost, recs, recPre, recInf, recPost, overheads, firstLineCount)

        assertTrue("Should complete all iterations", totals.size == iterations)
        assertTrue("Should detect text", firstLineCount > 0)
    }

    private fun printSpeedSummary(
        warmup: Int,
        iterations: Int,
        totals: List<Long>,
        dets: List<Long>,
        detPre: List<Long>,
        detInf: List<Long>,
        detPost: List<Long>,
        recs: List<Long>,
        recPre: List<Long>,
        recInf: List<Long>,
        recPost: List<Long>,
        overheads: List<Long>,
        lineCount: Int,
    ) {
        val bar = "+-----------------------------+----------+----------+----------+----------+"
        val header = "| Stage                       |  Mean ms |    Stdev |       P90 |    Min ms |"
        fun row(label: String, samples: List<Long>): String {
            if (samples.isEmpty()) return "| %-27s | %8s | %8s | %8s | %8s |".format(label, "-", "-", "-", "N/A")
            val mean = samples.map { it.toDouble() }.average()
            val variance = if (samples.size > 1) samples.map { (it - mean).pow(2) }.average() else 0.0
            val stdev = sqrt(variance)
            val sorted = samples.sorted()
            val p90 = sorted[(0.9 * (sorted.size - 1)).toInt()]
            val min = sorted.first()
            return "| %-27s | %8.2f | %8.2f | %8d | %8d |".format(label, mean, stdev, p90, min)
        }

        println("[OCRBenchmark] ")
        println("[OCRBenchmark] ╔════════════════════════════════════════════════════════════════════════════════╗")
        println("[OCRBenchmark] ║  PP-OCRv6 Speed Benchmark Results                                      ║")
        println("[OCRBenchmark] ╠════════════════════════════════════════════════════════════════════════════════╣")
        println("[OCRBenchmark] ║  Device: ${Build.MODEL}  |  OS: Android ${Build.VERSION.RELEASE}  |  Lines: $lineCount")
        println("[OCRBenchmark] ║  Cold load: ${ocr.coldLoadTimeMs}ms  |  Warmup: $warmup  |  Measured: $iterations")
        println("[OCRBenchmark] ╠════════════════════════════════════════════════════════════════════════════════╣")
        println("[OCRBenchmark] $bar")
        println("[OCRBenchmark] $header")
        println("[OCRBenchmark] $bar")
        println("[OCRBenchmark] ${row("Total pipeline", totals)}")
        println("[OCRBenchmark] $bar")
        println("[OCRBenchmark] ${row("  Detection (total)", dets)}")
        println("[OCRBenchmark] ${row("    Preprocess", detPre)}")
        println("[OCRBenchmark] ${row("    Inference", detInf)}")
        println("[OCRBenchmark] ${row("    Postprocess", detPost)}")
        println("[OCRBenchmark] ${row("  Recognition (total)", recs)}")
        println("[OCRBenchmark] ${row("    Preprocess", recPre)}")
        println("[OCRBenchmark] ${row("    Inference", recInf)}")
        println("[OCRBenchmark] ${row("    Postprocess", recPost)}")
        println("[OCRBenchmark] ${row("  Pipeline overhead", overheads)}")
        println("[OCRBenchmark] $bar")
        println("[OCRBenchmark] ╚════════════════════════════════════════════════════════════════════════════════╝")
        println("[OCRBenchmark] ")
    }

    // ========================================================================
    // Memory
    // ========================================================================

    @Test
    fun testMemoryBenchmark() {
        val imageBytes = loadBenchmarkImageBytes()
        val warmup = getInstrumentationArgument(ARG_WARMUP)?.toIntOrNull() ?: DEFAULT_WARMUP
        val iterations = getInstrumentationArgument(ARG_ITERATIONS)?.toIntOrNull() ?: DEFAULT_ITERATIONS

        runBlocking { repeat(warmup) { ocr.recognize(imageBytes) } }
        Runtime.getRuntime().gc()
        Thread.sleep(500)

        val before = Debug.MemoryInfo()
        Debug.getMemoryInfo(before)

        runBlocking { repeat(iterations) { ocr.recognize(imageBytes) } }

        val after = Debug.MemoryInfo()
        Debug.getMemoryInfo(after)

        val payload = JSONObject().apply {
            put("schemaVersion", 1)
            put("deviceModel", Build.MODEL)
            put("iterations", iterations)
            put("memoryBeforePssKb", before.totalPss)
            put("memoryAfterPssKb", after.totalPss)
            put("memoryDeltaPssKb", after.totalPss - before.totalPss)
            put("nativeBeforeKb", before.nativePss)
            put("nativeAfterKb", after.nativePss)
            put("dalvikBeforeKb", before.dalvikPss)
            put("dalvikAfterKb", after.dalvikPss)
        }

        saveAndPrint("memory_benchmark.json", payload)
        assertTrue("Memory should be measurable", after.totalPss > 0)
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private fun loadBenchmarkImageBytes(): ByteArray {
        val context = InstrumentationRegistry.getInstrumentation().context
        val stem = BenchmarkFixtures.defaultReferenceImageName.substringBeforeLast(".")
        val resId = context.resources.getIdentifier(stem, "raw", context.packageName)
        if (resId == 0) {
            throw IllegalStateException(
                "Missing ${BenchmarkFixtures.defaultReferenceImageName} " +
                        "(package=${context.packageName})"
            )
        }
        return context.resources.openRawResource(resId).use { it.readBytes() }
    }

    data class TimingSummary(val mean: Double, val stdev: Double, val p90: Double) {
        fun toJson() = JSONObject().apply {
            put("mean", mean)
            put("stdev", stdev)
            put("p90", p90)
        }
    }

    private fun summarizeMs(samples: List<Long>): TimingSummary {
        if (samples.isEmpty()) return TimingSummary(0.0, 0.0, 0.0)
        val doubles = samples.map { it.toDouble() }
        val mean = doubles.average()
        val variance = if (doubles.size > 1)
            doubles.map { (it - mean).pow(2) }.average()
        else 0.0
        val stdev = sqrt(variance)
        val sorted = doubles.sorted()
        val p90Idx = (0.9 * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return TimingSummary(mean, stdev, sorted[p90Idx])
    }

    private fun summarizeShapes(shapes: List<List<Int>>): JSONArray {
        val counts = mutableMapOf<String, Int>()
        shapes.forEach { shape ->
            val key = shape.joinToString(",")
            counts[key] = (counts[key] ?: 0) + 1
        }
        return JSONArray().apply {
            counts.entries.sortedByDescending { it.value }.forEach { (shape, count) ->
                put(JSONObject().apply {
                    put("shape", JSONArray(shape.split(",").map { it.toInt() }))
                    put("count", count)
                })
            }
        }
    }

    private fun buildExportPayload(result: OCRRunResult): JSONObject {
        val items = JSONArray().apply {
            result.results.forEach { r ->
                put(JSONObject().apply {
                    put("polygon", JSONArray().apply {
                        r.box.points.forEach { pt ->
                            put(JSONArray(listOf(pt.x.toInt(), pt.y.toInt())))
                        }
                    })
                    put("text", r.text)
                    put("score", r.confidence.toDouble())
                })
            }
        }
        return JSONObject().apply {
            put("schemaVersion", 1)
            put("source", "android_ocr_demo")
            put("items", items)
        }
    }
}
