// For a detailed explanation regarding each configuration property and type check, visit:
// https://jestjs.io/docs/en/configuration.html

module.exports = {
    preset: 'jest-puppeteer',
    // All imported modules in your tests should be mocked automatically
    // automock: false,

    // Automatically clear mock calls and instances between every test
    clearMocks: true,

    // An object that configures minimum threshold enforcement for coverage results
    // coverageThreshold: undefined,

    // A set of global variables that need to be available in all test environments
    globals: {
        PATH: 'http://localhost:9811'
    },

    // The maximum amount of workers used to run your tests. Can be specified as % or a number. E.g. maxWorkers: 10% will use 10% of your CPU amount + 1 as the maximum worker number. maxWorkers: 2 will use a maximum of 2 workers.
    // maxWorkers: "50%",

    // An array of directory names to be searched recursively up from the requiring module's location
    // moduleDirectories: [
    //   "node_modules"
    // ],

    // An array of file extensions your modules use
    moduleFileExtensions: [
        'js',
        'json',
        'jsx',
        'ts',
        'tsx',
        'node'
    ],


    // The root directory that Jest should scan for tests and modules within
    // rootDir: undefined,

    // A list of paths to directories that Jest should use to search for files in
    roots: [
        '<rootDir>'
    ],

    // Allows you to use a custom runner instead of Jest's default test runner
    // runner: "jest-runner",

    // The paths to modules that run some code to configure or set up the testing environment before each test
    // setupFiles: [],

    // A list of paths to modules that run some code to configure or set up the testing framework before each test
    // setupFilesAfterEnv: [],

    // The number of seconds after which a test is considered as slow and reported as such in the results.
    // slowTestThreshold: 5,

    // A list of paths to snapshot serializer modules Jest should use for snapshot testing
    // snapshotSerializers: [],

    // The test environment that will be used for testing
    // testEnvironment: 'jsdom',

    // Options that will be passed to the testEnvironment
    // testEnvironmentOptions: {},

    // An array of regexp pattern strings that are matched against all test paths, matched tests are skipped
    testPathIgnorePatterns: [
        '/node_modules/'
    ],

    // The regexp pattern or array of patterns that Jest uses to detect test files
    testRegex: '.(.+)\\.test\\.(js|ts)$',

    // This option allows the use of a custom results processor
    // testResultsProcessor: undefined,

    // This option allows use of a custom test runner
    // testRunner: "jest-circus/runner",

    // This option sets the URL for the jsdom environment. It is reflected in properties such as location.href
    testURL: 'http://localhost:9898/',

    // Setting this value to "fake" allows the use of fake timers for functions such as "setTimeout"
    // timers: "real",

    // A map from regular expressions to paths to transformers
    transform: {
        '^.+\\.js$': 'babel-jest'
    },

    // An array of regexp pattern strings that are matched against all source file paths, matched files will skip transformation
    transformIgnorePatterns: [
        '/node_modules/',
        '\\.pnp\\.[^\\/]+$'
    ],

    // An array of regexp pattern strings that are matched against all modules before the module loader will automatically return a mock for them
    // unmockedModulePathPatterns: undefined,

    // Indicates whether each individual test should be reported during the run
    verbose: true,

    // An array of regexp patterns that are matched against all source file paths before re-running tests in watch mode
    // watchPathIgnorePatterns: [],

    // Whether to use watchman for file crawling
    // watchman: true,
    testTimeout: 50000
};
