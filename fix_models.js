const fs = require('fs');

function fixModel(path) {
  console.log(`Fixing ${path}...`);
  try {
    let raw = fs.readFileSync(path, 'utf8');
    
    // Fix SiLU activation
    raw = raw.replace(/"silu"/g, '"swish"');
    
    let obj = JSON.parse(raw);

    function fix(o) {
      if (Array.isArray(o)) return o.map(fix);
      if (o !== null && typeof o === 'object') {
        
        // Brute-force the nodeData array format
        if (o.inbound_nodes) {
          o.inbound_nodes = o.inbound_nodes.map(node => {
            // If it's already the correct TFJS format [[ ... ]], leave it
            if (Array.isArray(node) && Array.isArray(node[0])) return node;
            
            // If it's the corrupted Keras 3 object format, map it perfectly
            if (!Array.isArray(node) && node.args) {
              let mapped = [];
              node.args.forEach(arg => {
                if (arg && arg.config && arg.config.keras_history) {
                  mapped.push([
                    arg.config.keras_history[0],       // Connected layer name
                    arg.config.keras_history[1] || 0,  // Node index
                    arg.config.keras_history[2] || 0,  // Tensor index
                    node.kwargs || {}                  // Kwargs
                  ]);
                }
              });
              return mapped.length > 0 ? mapped : [];
            }
            // Absolute fallback to prevent the nodeData crash
            return [];
          });
        }
        
        if (o.batch_shape) {
          o.batch_input_shape = o.batch_shape;
          delete o.batch_shape;
        }
        for (let key in o) o[key] = fix(o[key]);
      }
      return o;
    }

    let fixed = fix(obj);
    fs.writeFileSync(path, JSON.stringify(fixed, null, 2));
    console.log(`✅ Successfully fixed ${path}!`);
  } catch (e) {
    console.log(`❌ Error fixing ${path}:`, e.message);
  }
}

fixModel('./assets/images/models/beach/model.json');
fixModel('./assets/images/models/grain/model.json');