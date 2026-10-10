import { chromium } from "playwright";

const VIEWPORTS = [
  // Desktop
  { name: "desktop-1920", width: 1920, height: 1080, type: "desktop" },
  { name: "desktop-1440", width: 1440, height: 900, type: "desktop" },
  { name: "laptop-1280", width: 1280, height: 800, type: "desktop" },
  { name: "small-laptop-1024", width: 1024, height: 768, type: "desktop" },
  // Mobile / Tablet
  { name: "tablet-768", width: 768, height: 1024, type: "mobile" },
  { name: "mobile-430", width: 430, height: 932, type: "mobile" },
  { name: "mobile-390", width: 390, height: 844, type: "mobile" },
  { name: "mobile-375", width: 375, height: 812, type: "mobile" },
  { name: "mobile-360", width: 360, height: 800, type: "mobile" },
];

const PAGES = ["overview", "twin", "whatif", "comparison", "history", "settings"];

async function run() {
  console.log("=== Launching Headless Chrome via Playwright ===");
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });

  const results = [];
  const errors = [];

  for (const vp of VIEWPORTS) {
    console.log(`\n--- Testing Viewport: ${vp.name} (${vp.width}x${vp.height}) ---`);
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push({ viewport: vp.name, error: msg.text() });
      }
    });

    page.on("pageerror", (err) => {
      errors.push({ viewport: vp.name, pageError: err.message });
    });

    try {
      await page.goto("http://localhost:5173", { waitUntil: "networkidle", timeout: 15000 });
    } catch (e) {
      await page.goto("http://localhost:5173", { waitUntil: "domcontentloaded", timeout: 15000 });
    }

    // Wait 1s for chart animations & initial data
    await page.waitForTimeout(1000);

    for (const pageId of PAGES) {
      // Navigate to page
      if (vp.type === "mobile") {
        // Open drawer if not already open
        const isDrawerOpen = await page.evaluate(() => {
          const s = document.querySelector(".sidebar");
          return s ? s.classList.contains("mobile-open") : false;
        });
        if (!isDrawerOpen) {
          const toggleBtn = page.locator(".sidebar-toggle-btn");
          if (await toggleBtn.isVisible()) {
            await toggleBtn.click();
            await page.waitForTimeout(300);
          }
        }
      }

      // Click nav item
      const navBtn = page.locator(`.sidebar-nav button.sidebar-item`).filter({
        hasText: new RegExp(
          pageId === "overview" ? "Overview" :
          pageId === "twin" ? "My Digital Twin" :
          pageId === "whatif" ? "What-If Lab" :
          pageId === "comparison" ? "Model Benchmark" :
          pageId === "history" ? "History" : "Settings",
          "i"
        ),
      });

      if (await navBtn.isVisible()) {
        await navBtn.click();
        await page.waitForTimeout(600);
      }

      // Check horizontal overflow
      const overflow = await page.evaluate(() => {
        const docW = document.documentElement.scrollWidth;
        const winW = window.innerWidth;
        const bodyW = document.body.scrollWidth;
        // check elements causing overflow if any
        let maxElem = null;
        let maxElemW = 0;
        if (docW > winW) {
          document.querySelectorAll("*").forEach((el) => {
            const rect = el.getBoundingClientRect();
            if (rect.right > winW + 1 && rect.width > maxElemW) {
              maxElemW = rect.width;
              maxElem = {
                tag: el.tagName,
                cls: el.className,
                id: el.id,
                right: rect.right,
                width: rect.width,
              };
            }
          });
        }
        return {
          hasOverflow: docW > winW || bodyW > winW,
          docW,
          winW,
          bodyW,
          diff: Math.max(docW - winW, bodyW - winW),
          maxElem,
        };
      });

      console.log(`  [${vp.name}] Page: ${pageId} | Overflow: ${overflow.hasOverflow ? `FAIL (+${overflow.diff}px)` : "PASS"}`);
      if (overflow.hasOverflow && overflow.maxElem) {
        console.log(`    Offending element:`, JSON.stringify(overflow.maxElem));
      }

      results.push({
        viewport: vp.name,
        width: vp.width,
        page: pageId,
        overflow: overflow.hasOverflow,
        diff: overflow.diff,
        maxElem: overflow.maxElem,
      });
    }

    await context.close();
  }

  await browser.close();

  console.log("\n=== SUMMARY OF TESTS ===");
  const fails = results.filter((r) => r.overflow);
  console.log(`Total checks: ${results.length}`);
  console.log(`Passed checks: ${results.length - fails.length}`);
  console.log(`Failed checks: ${fails.length}`);
  if (fails.length > 0) {
    console.log("Failed details:", JSON.stringify(fails, null, 2));
  }
  if (errors.length > 0) {
    console.log("Console/Page errors:", JSON.stringify(errors, null, 2));
  }
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
