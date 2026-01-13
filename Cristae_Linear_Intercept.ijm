/*
 * ============================================================================
 * MITOCHONDRIAL CRISTAE DENSITY - LINEAR INTERCEPT METHOD
 * ============================================================================
 * 
 * Measures: Cristae Density = # cristae intercepts / mitochondria area
 * 
 * PHASE 1 (F1): Batch prep - Crop ROIs → Weka classifier → Save probability maps
 * PHASE 2 (F2): Interactive analysis - Draw line, count crossings, manual correction
 * 
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

var PADDING = 20;
var PROB_THRESHOLD = 128;        // For 8-bit (0-255)
var PROB_THRESHOLD_32 = 0.5;     // For 32-bit (0-1)
var MIN_PEAK_WIDTH = 2;
var MIN_PEAK_DISTANCE = 3;
var OUTPUT_FOLDER = "";
var CLASSIFIER_PATH = "";

// ============================================================================
// PHASE 1: BATCH PREPARATION
// ============================================================================

macro "Phase 1 - Batch Prep [F1]" {
    requires("1.53");
    
    if (nImages < 1) exit("Open your EM image first.");
    if (roiManager("count") < 1) exit("Load ROIs into ROI Manager first.");
    
    sourceID = getImageID();
    sourceImage = getTitle();
    
    Dialog.create("Phase 1: Batch Preparation");
    Dialog.addFile("Weka Classifier (.model):", CLASSIFIER_PATH);
    Dialog.addDirectory("Output Folder:", getDirectory("home") + "CristaeAnalysis/");
    Dialog.addNumber("Padding (px):", PADDING);
    Dialog.show();
    
    CLASSIFIER_PATH = Dialog.getString();
    OUTPUT_FOLDER = Dialog.getString();
    PADDING = Dialog.getNumber();
    
    if (!File.exists(CLASSIFIER_PATH)) exit("Classifier not found.");
    
    // Create folders
    if (!File.exists(OUTPUT_FOLDER)) File.makeDirectory(OUTPUT_FOLDER);
    rawFolder = OUTPUT_FOLDER + "raw_crops/";
    probFolder = OUTPUT_FOLDER + "probability_maps/";
    metaFolder = OUTPUT_FOLDER + "metadata/";
    if (!File.exists(rawFolder)) File.makeDirectory(rawFolder);
    if (!File.exists(probFolder)) File.makeDirectory(probFolder);
    if (!File.exists(metaFolder)) File.makeDirectory(metaFolder);
    
    roiManager("Save", OUTPUT_FOLDER + "ROI_set.zip");
    
    print("\\Clear");
    print("=== PHASE 1: BATCH PREPARATION ===");
    print("Source: " + sourceImage);
    print("ROIs: " + roiManager("count"));
    print("");
    
    nROIs = roiManager("count");
    startTime = getTime();
    processedCount = 0;

    for (i = 0; i < nROIs; i++) {
        // Force garbage collection every 50 ROIs
        if (i > 0 && i % 50 == 0) {
            print("-- Clearing memory (" + i + "/" + nROIs + ") --");
            run("Collect Garbage");
            wait(300);
            run("Collect Garbage");
            wait(500);
        }
        
        selectImage(sourceID);
        roiManager("select", i);
        
        roiName = Roi.getName();
        if (roiName == "") roiName = "Mito_" + IJ.pad(i + 1, 3);
        
        // Skip if already processed
        probPath = probFolder + roiName + "_prob.tif";
        rawPath = rawFolder + roiName + "_raw.tif";
        
        // Debug: show what we're checking
        //print("Checking: " + probPath);
        //print("  Raw exists: " + File.exists(rawPath));
        //print("  Prob exists: " + File.exists(probPath));
        
        if (File.exists(probPath) && File.exists(rawPath)) {
            print("Skipping " + (i + 1) + "/" + nROIs + ": " + roiName + " (already done)");
            continue;
        }
        
        print("Processing " + (i + 1) + "/" + nROIs + ": " + roiName);
        processedCount++;
        
        Roi.getBounds(rx, ry, rw, rh);
        if (rw == 0 || rh == 0) {
            print("  ERROR: Invalid ROI bounds");
            continue;
        }
        
        imgW = getWidth();
        imgH = getHeight();
        
        // Crop bounds with padding
        x1 = rx - PADDING;
        if (x1 < 0) x1 = 0;
        y1 = ry - PADDING;
        if (y1 < 0) y1 = 0;
        x2 = rx + rw + PADDING;
        if (x2 > imgW) x2 = imgW;
        y2 = ry + rh + PADDING;
        if (y2 > imgH) y2 = imgH;
        
        offsetX = rx - x1;
        offsetY = ry - y1;
        cropW = x2 - x1;
        cropH = y2 - y1;
        
        // Crop
        selectImage(sourceID);
        run("Select None");
        makeRectangle(x1, y1, cropW, cropH);
        run("Duplicate...", "title=[" + roiName + "_crop]");
        cropID = getImageID();
        
        // Save raw
        saveAs("Tiff", rawFolder + roiName + "_raw.tif");
        rename(roiName + "_crop");
        
        // Run Weka with retry logic
        wekaSuccess = false;
        maxRetries = 2;
        
        for (attempt = 0; attempt < maxRetries && !wekaSuccess; attempt++) {
            if (attempt > 0) {
                print("  Retry attempt " + (attempt + 1) + "...");
                run("Collect Garbage");
                wait(1000);
            }
            
            selectImage(cropID);  // Make sure crop is selected
            run("Trainable Weka Segmentation");
            wait(2000);  // Give Weka time to initialize

            wekaTitle = findWindowByPattern("Trainable Weka");
            if (wekaTitle == "") {
                print("  ERROR: Weka failed to open");
                continue;
            }
            
            selectWindow(wekaTitle);
            
            // Load classifier
            print("  Loading classifier...");
            logBefore = getInfo("log");
            logLenBefore = lengthOf(logBefore);
            call("trainableSegmentation.Weka_Segmentation.loadClassifier", CLASSIFIER_PATH);

            // Poll for classifier to finish loading (log shows "Loaded" when done)
            maxWait = 10000;  // 10 second max (safety)
            pollInterval = 100;  // Check every 100ms
            waited = 0;
            classifierLoaded = false;
            while (waited < maxWait && !classifierLoaded) {
                wait(pollInterval);
                waited += pollInterval;
                logNow = getInfo("log");
                if (lengthOf(logNow) > logLenBefore) {
                    newLogContent = substring(logNow, logLenBefore);
                    if (indexOf(newLogContent, "Loaded") >= 0) {
                        classifierLoaded = true;
                    }
                }
            }
            if (!classifierLoaded) {
                print("  WARNING: Classifier load timeout, restarting Weka...");
                wekaTitle = findWindowByPattern("Trainable Weka");
                if (wekaTitle != "") {
                    selectWindow(wekaTitle);
                    close();
                }
                continue;  // Retry from the beginning of the attempt loop
            }
            
            // Apply classifier to get probability
            print("  Applying classifier...");
            call("trainableSegmentation.Weka_Segmentation.getProbability");
            
            // Poll for probability window (classifier may take variable time)
            maxWaitTime = 30000;  // 30 seconds max
            pollInterval = 500;   // Check every 500ms
            waited = 0;
            probWindow = "";
            
            while (waited < maxWaitTime && probWindow == "") {
                wait(pollInterval);
                waited += pollInterval;
                probWindow = findWindowByPattern("Probability");
                if (waited % 5000 == 0 && probWindow == "") {
                    print("    Still waiting for classifier... (" + (waited/1000) + "s)");
                }
            }
            
            // Check if probability map was created
            if (probWindow != "") {
                wekaSuccess = true;
            } else {
                print("  WARNING: No probability map after " + (maxWaitTime/1000) + "s - closing Weka and retrying");
                // Close Weka window before retry
                wekaTitle = findWindowByPattern("Trainable Weka");
                if (wekaTitle != "") {
                    selectWindow(wekaTitle);
                    close();
                    wait(200);
                }
            }
        }
        
        if (!wekaSuccess) {
            print("  ERROR: Weka failed after " + maxRetries + " attempts, skipping ROI");
            closeAllExcept(sourceID);
            continue;
        }
        
        // Save probability map
        probWindow = findWindowByPattern("Probability");
        if (probWindow != "") {
            selectWindow(probWindow);
            getDimensions(pw, ph, pChannels, pSlices, pFrames);
            if (pChannels > 1) {
                run("Duplicate...", "title=[" + roiName + "_prob] channels=1");
            } else {
                run("Duplicate...", "title=[" + roiName + "_prob]");
            }
            saveAs("Tiff", probFolder + roiName + "_prob.tif");
            close();
        } else {
            print("  WARNING: No probability map generated");
        }
        
        // Save metadata
        metaFile = File.open(metaFolder + roiName + "_meta.txt");
        print(metaFile, "offset_x=" + offsetX);
        print(metaFile, "offset_y=" + offsetY);
        print(metaFile, "roi_width=" + rw);
        print(metaFile, "roi_height=" + rh);
        print(metaFile, "roi_index=" + i);
        File.close(metaFile);
        
        // Cleanup - close Weka to free memory
        // Close the Weka segmentation window
        wekaTitle = findWindowByPattern("Trainable Weka");
        if (wekaTitle != "") {
            selectWindow(wekaTitle);
            close();
        }

        // Close any remaining probability windows
        probWindow = findWindowByPattern("Probability");
        while (probWindow != "") {
            selectWindow(probWindow);
            close();
            probWindow = findWindowByPattern("Probability");
        }

        // Close any remaining Weka windows (duplicates can happen)
        wekaTitle = findWindowByPattern("Trainable Weka");
        while (wekaTitle != "") {
            selectWindow(wekaTitle);
            close();
            wekaTitle = findWindowByPattern("Trainable Weka");
        }
        
        // Close all images except source
        closeAllExcept(sourceID);
        
        // Force garbage collection after EVERY ROI to prevent memory buildup
        run("Collect Garbage");
        
        elapsed = (getTime() - startTime) / 1000;
        avgTime = elapsed / processedCount;
        remaining = avgTime * (nROIs - i - 1);
        print("  Done. ~" + round(remaining) + "s remaining");
    }
    
    print("\n=== PREP COMPLETE ===");
    print("Run Phase 2 (F2) for analysis.");
    showMessage("Prep Complete", "Processed " + nROIs + " ROIs.\nRun Phase 2 (F2) for analysis.");
}

// ============================================================================
// PHASE 2: INTERACTIVE ANALYSIS
// ============================================================================

macro "Phase 2 - Analyze [F2]" {
    requires("1.53");
    
    Dialog.create("Phase 2: Analysis");
    Dialog.addDirectory("Prep Folder:", OUTPUT_FOLDER);
    Dialog.addNumber("Threshold:", PROB_THRESHOLD);
    Dialog.addNumber("Min Peak Width:", MIN_PEAK_WIDTH);
    Dialog.addNumber("Min Peak Distance:", MIN_PEAK_DISTANCE);
    Dialog.show();
    
    OUTPUT_FOLDER = Dialog.getString();
    PROB_THRESHOLD = Dialog.getNumber();
    MIN_PEAK_WIDTH = Dialog.getNumber();
    MIN_PEAK_DISTANCE = Dialog.getNumber();
    
    rawFolder = OUTPUT_FOLDER + "raw_crops/";
    probFolder = OUTPUT_FOLDER + "probability_maps/";
    metaFolder = OUTPUT_FOLDER + "metadata/";
    resultsPath = OUTPUT_FOLDER + "results_autosave.csv";
    
    if (!File.exists(probFolder)) exit("Run Phase 1 first.");
    
    // Get mito list
    probFiles = getFileList(probFolder);
    mitoList = newArray(0);
    for (fi = 0; fi < probFiles.length; fi++) {
        if (endsWith(probFiles[fi], "_prob.tif")) {
            mitoList = Array.concat(mitoList, replace(probFiles[fi], "_prob.tif", ""));
        }
    }
    nMitos = mitoList.length;
    if (nMitos == 0) exit("No probability maps found.");
    
    // Load ROIs
    roiPath = OUTPUT_FOLDER + "ROI_set.zip";
    if (File.exists(roiPath)) {
        roiManager("reset");
        roiManager("Open", roiPath);
    }
    
    // Check for existing results to resume
    startFrom = 0;
    if (File.exists(resultsPath)) {
        Dialog.create("Resume?");
        Dialog.addMessage("Found saved progress: " + resultsPath);
        Dialog.addCheckbox("Load previous results and continue", true);
        Dialog.show();
        
        if (Dialog.getCheckbox()) {
            // Load existing results into Results table
            run("Clear Results");
            Table.open(resultsPath);
            // Rename to Results if needed
            if (isOpen("results_autosave.csv")) {
                selectWindow("results_autosave.csv");
                Table.rename("results_autosave.csv", "Results");
            }
            startFrom = nResults;
        } else {
            run("Clear Results");
        }
    } else {
        run("Clear Results");
    }
    
    print("\\Clear");
    print("=== PHASE 2: ANALYSIS ===");
    print("Mitochondria: " + nMitos);
    print("Progress auto-saved to: " + resultsPath);
    print("");
    
    // Show start dialog with correct default
    Dialog.create("Start");
    Dialog.addNumber("Start from ROI #:", startFrom + 1);
    Dialog.show();
    startFrom = Dialog.getNumber() - 1;
    if (startFrom < 0) startFrom = 0;
    if (startFrom >= nMitos) startFrom = 0;
    
    // Main loop
    for (i = startFrom; i < nMitos; i++) {
        roiName = mitoList[i];
        print("--- " + (i+1) + "/" + nMitos + ": " + roiName + " ---");
        
        // Load metadata
        metaPath = metaFolder + roiName + "_meta.txt";
        offsetX = PADDING;
        offsetY = PADDING;
        roiW = 0;
        roiH = 0;
        roiIndex = i;
        
        if (File.exists(metaPath)) {
            meta = File.openAsString(metaPath);
            lines = split(meta, "\n");
            for (mi = 0; mi < lines.length; mi++) {
                if (startsWith(lines[mi], "offset_x=")) offsetX = parseInt(replace(lines[mi], "offset_x=", ""));
                if (startsWith(lines[mi], "offset_y=")) offsetY = parseInt(replace(lines[mi], "offset_y=", ""));
                if (startsWith(lines[mi], "roi_width=")) roiW = parseInt(replace(lines[mi], "roi_width=", ""));
                if (startsWith(lines[mi], "roi_height=")) roiH = parseInt(replace(lines[mi], "roi_height=", ""));
                if (startsWith(lines[mi], "roi_index=")) roiIndex = parseInt(replace(lines[mi], "roi_index=", ""));
            }
        }
        
        // Open images
        rawPath = rawFolder + roiName + "_raw.tif";
        probPath = probFolder + roiName + "_prob.tif";
        if (!File.exists(rawPath) || !File.exists(probPath)) {
            print("  Missing files, skipping.");
            continue;
        }
        
        open(rawPath);
        rawID = getImageID();
        open(probPath);
        probID = getImageID();
        
        if (bitDepth() == 32) {
            thresh = PROB_THRESHOLD_32;
        } else {
            thresh = PROB_THRESHOLD;
        }
        
        // Get ROI area
        roiArea = roiW * roiH;
        if (roiArea == 0) {
            roiArea = (getWidth() - 2*PADDING) * (getHeight() - 2*PADDING);
            if (roiArea <= 0) roiArea = getWidth() * getHeight();
        }
        if (roiManager("count") > roiIndex) {
            selectImage(rawID);
            roiManager("select", roiIndex);
            Roi.move(offsetX, offsetY);
            getStatistics(measuredArea);
            if (measuredArea > 0) roiArea = measuredArea;
        }
        
        // Get image dimensions and scale window size
        selectImage(rawID);
        imgW = getWidth();
        imgH = getHeight();
        
        // Scale to fit nicely - minimum 300, maximum 500
        scale = 2.5;  // magnification factor
        winW = imgW * scale;
        winH = imgH * scale;
        if (winW < 300) winW = 300;
        if (winH < 300) winH = 300;
        if (winW > 500) winW = 500;
        if (winH > 500) winH = 500;
        
        // Position windows
        selectImage(rawID);
        setLocation(20, 50, winW, winH);
        run("Enhance Contrast", "saturated=0.3");
        selectImage(probID);
        setLocation(40 + winW, 50, winW, winH);
        
        // Draw ROI outline
        if (roiManager("count") > roiIndex) {
            selectImage(rawID);
            roiManager("select", roiIndex);
            Roi.move(offsetX, offsetY);
            Overlay.addSelection("green", 1);
            Overlay.show();
            selectImage(probID);
            roiManager("select", roiIndex);
            Roi.move(offsetX, offsetY);
            Overlay.addSelection("green", 1);
            Overlay.show();
        }
        
        // Interaction
        selectImage(rawID);
        run("Select None");
        setTool("line");
        
        result = analyzeROI(rawID, probID, roiName, i, nMitos, thresh, roiIndex, offsetX, offsetY);
        // result: [action, count, lineLength]
        // action: "accept", "skip", "back", "quit"
        
        action = result[0];
        finalCount = result[1];
        lineLength = result[2];
        
        // Handle result
        if (action == "quit") {
            print("  QUIT - progress saved");
            saveAs("Results", resultsPath);
            closeImages(rawID, probID);
            break;
        } else if (action == "back") {
            print("  GO BACK");
            closeImages(rawID, probID);
            if (nResults > 0) IJ.deleteRows(nResults - 1, nResults - 1);
            saveAs("Results", resultsPath);
            i = i - 2;
            continue;
        } else if (action == "skip") {
            print("  SKIPPED");
            recordResult(roiName, roiArea, 0, 0, 0);
            saveAs("Results", resultsPath);
            closeImages(rawID, probID);
            continue;
        } else {
            // Accept
            density = (finalCount / roiArea) * 1000;
            print("  Count: " + finalCount + ", Density: " + d2s(density, 3) + "/1000px²");
            recordResult(roiName, roiArea, finalCount, lineLength, density);
            saveAs("Results", resultsPath);
            closeImages(rawID, probID);
        }
    }
    
    // Summary
    print("\n=== COMPLETE ===");
    if (nResults > 0) {
        validCount = 0;
        sum = 0;
        for (si = 0; si < nResults; si++) {
            d = getResult("Density_per_1000px2", si);
            if (d > 0) {
                sum += d;
                validCount++;
            }
        }
        if (validCount > 0) {
            print("Valid: " + validCount + "/" + nResults);
            print("Mean density: " + d2s(sum/validCount, 3) + "/1000px²");
        }
    }
    showMessage("Complete", "Analyzed " + nResults + " ROIs.\nPress E to export CSV.");
}

// ============================================================================
// INTERACTION FUNCTION
// ============================================================================

function analyzeROI(rawID, probID, roiName, idx, total, thresh, roiIndex, offsetX, offsetY) {
    /*
     * Handle user interaction for one ROI
     * Returns: [action, count, lineLength]
     */
    
    totalCount = 0;
    totalLineLen = 0;
    lineNum = 0;
    
    while (true) {
        // Dialog to draw line
        selectImage(rawID);
        setTool("line");
        
        Dialog.createNonBlocking("[" + (idx+1) + "/" + total + "] " + roiName);
        if (lineNum == 0) {
            Dialog.addMessage("Draw line across cristae, then click OK");
        } else {
            Dialog.addMessage("Running total: " + totalCount + " (" + lineNum + " lines)\nDraw another line");
        }
        Dialog.addChoice("Action:", newArray("OK", "SKIP", "GO BACK", "QUIT"), "OK");
        Dialog.show();
        
        firstChoice = Dialog.getChoice();
        
        if (firstChoice == "SKIP") {
            return newArray("skip", 0, 0);
        } else if (firstChoice == "GO BACK") {
            return newArray("back", 0, 0);
        } else if (firstChoice == "QUIT") {
            return newArray("quit", 0, 0);
        }
        
        // Check if line was drawn
        selectImage(rawID);
        if (selectionType() != 5) {
            if (lineNum == 0) {
                return newArray("skip", 0, 0);
            } else {
                // No new line - finish with what we have
                return newArray("accept", totalCount, totalLineLen);
            }
        }
        
        // Get line coordinates
        getLine(lx1, ly1, lx2, ly2, lw);
        lineLen = sqrt(pow(lx2-lx1, 2) + pow(ly2-ly1, 2));
        
        // Get profile and count
        selectImage(probID);
        makeLine(lx1, ly1, lx2, ly2);
        profile = getProfile();
        result = countThresholdCrossings(profile, thresh);
        autoCount = result[0];
        peakPositions = result[1];
        
        // Show preview
        drawMarkersAdditive(rawID, lx1, ly1, lx2, ly2, peakPositions);
        drawMarkersAdditive(probID, lx1, ly1, lx2, ly2, peakPositions);
        
        selectImage(rawID);
        makeLine(lx1, ly1, lx2, ly2);
        
        // Confirm dialog
        Dialog.create("[" + (idx+1) + "/" + total + "] Confirm");
        if (lineNum == 0) {
            Dialog.addMessage("Detected: " + autoCount + " cristae");
            Dialog.addNumber("Final count:", autoCount);
        } else {
            newTotal = totalCount + autoCount;
            Dialog.addMessage("This line: " + autoCount + "\nNew total: " + newTotal);
            Dialog.addNumber("Final count:", newTotal);
        }
        Dialog.addChoice("Action:", newArray("ACCEPT", "ADD LINE", "REDRAW", "SKIP", "GO BACK", "QUIT"), "ACCEPT");
        Dialog.show();
        
        finalCount = Dialog.getNumber();
        choice = Dialog.getChoice();
        
        if (choice == "ACCEPT") {
            return newArray("accept", finalCount, totalLineLen + lineLen);
        } else if (choice == "ADD LINE") {
            // Add this line's count and continue for more
            totalCount = finalCount;
            totalLineLen = totalLineLen + lineLen;
            lineNum++;
            selectImage(rawID);
            run("Select None");
            // Loop back to draw another line
        } else if (choice == "REDRAW") {
            // Clear overlays and start fresh
            selectImage(rawID);
            Overlay.remove();
            selectImage(probID);
            Overlay.remove();
            // Redraw ROI outline
            if (roiManager("count") > roiIndex) {
                selectImage(rawID);
                roiManager("select", roiIndex);
                Roi.move(offsetX, offsetY);
                Overlay.addSelection("green", 1);
                Overlay.show();
                selectImage(probID);
                roiManager("select", roiIndex);
                Roi.move(offsetX, offsetY);
                Overlay.addSelection("green", 1);
                Overlay.show();
            }
            selectImage(rawID);
            run("Select None");
            // Reset totals
            totalCount = 0;
            totalLineLen = 0;
            lineNum = 0;
        } else if (choice == "SKIP") {
            return newArray("skip", 0, 0);
        } else if (choice == "GO BACK") {
            return newArray("back", 0, 0);
        } else if (choice == "QUIT") {
            return newArray("quit", 0, 0);
        }
    }
}

function drawMarkersAdditive(imgID, x1, y1, x2, y2, peakPositions) {
    selectImage(imgID);
    
    if (peakPositions != "") {
        dx = x2 - x1;
        dy = y2 - y1;
        positions = split(peakPositions, ",");
        
        for (k = 0; k < positions.length; k++) {
            t = parseFloat(positions[k]);
            mx = x1 + t * dx;
            my = y1 + t * dy;
            makeOval(mx - 3, my - 3, 6, 6);
            Overlay.addSelection("yellow", 2);
        }
    }
    
    makeLine(x1, y1, x2, y2);
    Overlay.addSelection("cyan", 1);
    Overlay.show();
    run("Select None");
}

function updateOverlays(rawID, probID, x1, y1, x2, y2, peakPositions, roiIndex, offsetX, offsetY) {
    // Clear overlays
    selectImage(rawID);
    Overlay.remove();
    selectImage(probID);
    Overlay.remove();
    
    // ROI outline on BOTH images
    if (roiManager("count") > roiIndex) {
        selectImage(rawID);
        roiManager("select", roiIndex);
        Roi.move(offsetX, offsetY);
        Overlay.addSelection("green", 1);
        
        selectImage(probID);
        roiManager("select", roiIndex);
        Roi.move(offsetX, offsetY);
        Overlay.addSelection("green", 1);
    }
    
    // Draw markers on both
    selectImage(rawID);
    drawMarkers(x1, y1, x2, y2, peakPositions);
    selectImage(probID);
    drawMarkers(x1, y1, x2, y2, peakPositions);
    
    // Restore line on both
    selectImage(rawID);
    makeLine(x1, y1, x2, y2);
    selectImage(probID);
    makeLine(x1, y1, x2, y2);
    selectImage(rawID);
}

function drawMarkers(x1, y1, x2, y2, peakPositions) {
    if (peakPositions == "") {
        makeLine(x1, y1, x2, y2);
        Overlay.addSelection("cyan", 1);
        Overlay.show();
        return;
    }
    
    dx = x2 - x1;
    dy = y2 - y1;
    positions = split(peakPositions, ",");
    
    for (k = 0; k < positions.length; k++) {
        t = parseFloat(positions[k]);
        mx = x1 + t * dx;
        my = y1 + t * dy;
        makeOval(mx - 3, my - 3, 6, 6);
        Overlay.addSelection("yellow", 2);
    }
    
    makeLine(x1, y1, x2, y2);
    Overlay.addSelection("cyan", 1);
    Overlay.show();
    run("Select None");
}

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

function countThresholdCrossings(profile, threshold) {
    n = profile.length;
    if (n < 3) return newArray(0, "");
    
    count = 0;
    positions = "";
    inPeak = false;
    peakStart = 0;
    lastPeakEnd = -MIN_PEAK_DISTANCE;
    
    for (idx = 0; idx < n; idx++) {
        above = (profile[idx] >= threshold);
        
        if (above && !inPeak) {
            if ((idx - lastPeakEnd) >= MIN_PEAK_DISTANCE) {
                inPeak = true;
                peakStart = idx;
            }
        } else if (!above && inPeak) {
            width = idx - peakStart;
            if (width >= MIN_PEAK_WIDTH) {
                center = (peakStart + idx - 1) / 2.0;
                count++;
                if (positions != "") positions += ",";
                positions += d2s(center / n, 4);
            }
            inPeak = false;
            lastPeakEnd = idx;
        }
    }
    
    // Peak at end
    if (inPeak) {
        width = n - peakStart;
        if (width >= MIN_PEAK_WIDTH) {
            center = (peakStart + n - 1) / 2.0;
            count++;
            if (positions != "") positions += ",";
            positions += d2s(center / n, 4);
        }
    }
    
    return newArray(count, positions);
}

function recordResult(name, area, count, lineLen, density) {
    row = nResults;
    setResult("Mito_Label", row, name);
    setResult("ROI_Area_px2", row, area);
    setResult("Cristae_Count", row, count);
    setResult("Line_Length_px", row, lineLen);
    setResult("Density_per_1000px2", row, density);
    updateResults();
}

function closeImages(id1, id2) {
    wait(20);
    selectImage(id1);
    Overlay.remove();
    wait(20);
    close();
    wait(20);
    selectImage(id2);
    Overlay.remove();
    wait(20);
    close();
}

function closeAllExcept(keepID) {
    wait(50);
    while (nImages > 1) {
        found = false;
        for (j = 1; j <= nImages; j++) {
            selectImage(j);
            if (getImageID() != keepID) {
                close();
                found = true;
                break;
            }
        }
        if (!found) break;
    }
}

function findWindowByPattern(pattern) {
    list = getList("window.titles");
    for (j = 0; j < list.length; j++) {
        if (indexOf(list[j], pattern) >= 0) return list[j];
    }
    list = getList("image.titles");
    for (j = 0; j < list.length; j++) {
        if (indexOf(list[j], pattern) >= 0) return list[j];
    }
    return "";
}


// ============================================================================
// UTILITIES
// ============================================================================

macro "Export CSV [e]" {
    if (nResults < 1) exit("No results.");
    path = File.saveDialog("Save CSV", "Cristae_Density.csv");
    if (path != "") saveAs("Results", path);
}

macro "Help [h]" {
    showMessage("Cristae Density Analysis",
        "F1: Phase 1 - Prep (crop ROIs, run Weka)\n" +
        "F2: Phase 2 - Analyze (draw lines, count)\n" +
        "E: Export results to CSV\n\n" +
        "During analysis:\n" +
        "- Draw line across cristae\n" +
        "- Yellow dots = detected crossings\n" +
        "- ADD LINE: draw additional lines\n" +
        "- REDRAW: clear and start over");
}
