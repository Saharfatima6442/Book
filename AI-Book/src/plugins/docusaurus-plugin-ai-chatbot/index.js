const path = require('path');

module.exports = function (context, options) {
  return {
    name: 'docusaurus-plugin-ai-chatbot',

    getClientModules() {
      return [path.resolve(__dirname, './src/client-modules/chatbot-fab')];
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': path.resolve(__dirname, '../../components/Chatbot'),
          },
        },
      };
    },
  };
};