// jest-puppeteer.config.js
module.exports = {
    launch: {
        headless: false,
        product: 'chrome'
    },
    browserContext: 'default',
    server: {
        command: 'python3 -m http.server 9811',
        port: 9811,
        launchTimeout: 10000,
        debug: true
    }
};
