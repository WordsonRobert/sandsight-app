const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Tell the bundler to stop ignoring our AI model files
config.resolver.assetExts.push('bin');

module.exports = config;